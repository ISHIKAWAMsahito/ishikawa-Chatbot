import logging
import uuid
import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, AsyncGenerator, Optional
from fastapi import Request

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from core.constants import PARAMS, AI_MESSAGES
from models.schemas import ChatQuery, ChatLogCreate
from services.llm import LLMService
from services.search import SearchService
from services.storage import StorageService
from services.chat_log import ChatLogService
from services.utils import (
    get_or_create_session_id,
    send_sse,
    log_context,
    ChatHistoryManager,
    format_references,
)
from services import prompts
from services.vectorize_logs import vectorize_chat_log

llm_service = LLMService()
search_service = SearchService(llm_service)
storage_service = StorageService()
history_manager = ChatHistoryManager(max_length=PARAMS["MAX_HISTORY_LENGTH"])


@traceable(name="Chat_Pipeline_Parent", run_type="chain")
async def enhanced_chat_logic(
    request: Request, chat_req: ChatQuery
) -> AsyncGenerator[str, None]:
    """
    RAGチャットロジック（FAQ優先・プライバシー保護対応版）
    - search_mode を ChatQuery から受け取り検索パイプラインに渡す
    - 自動ベクトル化バックグラウンドタスクは削除（次元数不一致エラー回避）
    """
    session_id = get_or_create_session_id(request)
    user_input = chat_req.question

    collection_name = getattr(chat_req, "collection", "student-knowledge-base")
    top_k = getattr(chat_req, "top_k", 5)
    embedding_model = getattr(chat_req, "embedding_model", "models/gemini-embedding-001")
    # ★ search_mode を取得（デフォルト: hybrid）
    search_mode = getattr(chat_req, "search_mode", "hybrid")

    run_tree = get_current_run_tree()

    # プライバシー保護: クエリ内容をログに出さない
    log_context(session_id, "Start processing query")

    JST = timezone(timedelta(hours=9), "JST")
    now = datetime.now(JST)
    current_date_str = now.strftime("%Y年%m月%d日")

    search_results = []
    is_faq_match = False
    ai_response_full = ""

    try:
        history_manager.add(session_id, "user", user_input)

        yield send_sse({"status_message": "🔍 質問を分析しています..."})

        # 1. クエリ拡張
        expanded_query = await search_service.expand_query(user_input)

        # 2. 検索パイプライン（search_mode を渡す）
        search_result_obj = await search_service.search(
            query=expanded_query,
            session_id=session_id,
            collection_name=collection_name,
            top_k=top_k,
            embedding_model=embedding_model,
            search_mode=search_mode,
        )
        search_results = search_result_obj.get("documents", [])
        is_faq_match = search_result_obj.get("is_faq_match", False)

        # ヒットしなかった場合、元クエリで再検索
        if not search_results and expanded_query != user_input:
            search_result_obj = await search_service.search(
                query=user_input,
                session_id=session_id,
                collection_name=collection_name,
                top_k=top_k,
                embedding_model=embedding_model,
                search_mode=search_mode,
            )
            search_results = search_result_obj.get("documents", [])
            is_faq_match = search_result_obj.get("is_faq_match", False)

        if not search_results:
            not_found_msg = AI_MESSAGES.get(
                "NOT_FOUND", "申し訳ありません。関連情報が見つかりませんでした。"
            )
            yield send_sse({"content": not_found_msg})
            history_manager.add(session_id, "assistant", not_found_msg)

            log_entry = ChatLogCreate(
                session_id=session_id,
                user_query=user_input,
                ai_response=not_found_msg,
                metadata={
                    "collection": collection_name,
                    "result": "not_found",
                    "top_k": top_k,
                    "search_mode": search_mode,
                },
            )
            # ★ 非同期タスクとして保存（ベクトル化は行わない）
            asyncio.create_task(_save_and_vectorize_log(log_entry))

            yield send_sse({"done": True, "feedback_id": str(uuid.uuid4())})
            return

        yield send_sse({"status_message": "✍️ 回答を生成しています..."})

        # 3. 回答生成
        chat_history = history_manager.get_history(session_id)

        context_parts = []
        for idx, doc in enumerate(search_results, 1):
            doc_content = doc.get("content", "")
            context_parts.append(f"<doc id='{idx}'>{doc_content}</doc>")
        context_str = "\n".join(context_parts)

        system_prompt_base = prompts.SYSTEM_GENERATION
        if is_faq_match:
            system_prompt_base += (
                "\n\n**重要: ユーザーの質問に完全に合致するFAQ資料が見つかりました。"
                "<doc id='1'>の内容を最優先し、その回答を正確に伝えてください。**"
            )

        try:
            full_system_prompt = system_prompt_base.format(
                context_text=context_str, current_date=current_date_str
            )
        except Exception:
            full_system_prompt = f"以下の情報を元に回答してください。\n{context_str}"

        async for chunk in llm_service.generate_response_stream(
            query=user_input,
            context_docs=search_results,
            history=chat_history,
            system_prompt=full_system_prompt,
        ):
            text_chunk = chunk if isinstance(chunk, str) else chunk.get("content", "")
            ai_response_full += text_chunk
            yield send_sse({"content": text_chunk})

        # 4. 参照元リスト
        references_text = format_references(search_results)
        if references_text:
            yield send_sse({"content": references_text})
            ai_response_full += "\n" + references_text

        # 5. 履歴保存
        history_manager.add(session_id, "assistant", ai_response_full)

        # 6. チャットログ保存と自動ベクトル化
        log_entry = ChatLogCreate(
            session_id=session_id,
            user_query=user_input,
            ai_response=ai_response_full,
            metadata={
                "collection": collection_name,
                "top_k": top_k,
                "is_faq_match": is_faq_match,
                "result": "success",
                "doc_count": len(search_results),
                "search_mode": search_mode,
            },
        )
        asyncio.create_task(_save_and_vectorize_log(log_entry))

        feedback_id = str(uuid.uuid4())
        yield send_sse({"done": True, "feedback_id": feedback_id})

    except Exception as e:
        log_context(session_id, f"Pipeline Error: {type(e).__name__}", "error", exc_info=True)
        if run_tree:
            run_tree.end(error=str(e))

        error_str = str(e)
        msg = AI_MESSAGES.get("SYSTEM_ERROR", "システムエラーが発生しました。")
        yield send_sse({"content": f"\n\n{msg}"})

    finally:
        log_context(session_id, "Response generation finished.")


async def _save_and_vectorize_log(log_entry: ChatLogCreate) -> None:
    """
    チャットログをDBに保存し、バックグラウンドで自動ベクトル化を行う。
    """
    try:
        data = log_entry.model_dump(mode="json")
        from core.database import db_client
        response = db_client.client.table("chat_logs").insert(data).execute()
        
        if response.data:
            log_id = response.data[0].get("id")
            logging.getLogger(__name__).info(f"Chat log saved. ID: {log_id}")
            
            # ★ 追加: 保存成功後に自動ベクトル化タスクを起動
            user_query = data.get("user_query", "")
            ai_response = data.get("ai_response", "")
            if user_query and ai_response:
                # vectorize_chat_log をバックグラウンドで実行
                asyncio.create_task(vectorize_chat_log(log_id, user_query, ai_response))
                
        else:
            logging.getLogger(__name__).warning("Chat log saved but no data returned.")
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to save and vectorize chat log: {e}", exc_info=True)


@traceable(name="Feedback_Analysis_Job", run_type="chain")
async def analyze_feedback_trends(
    logs: List[Dict[str, Any]]
) -> AsyncGenerator[str, None]:
    """フィードバック分析用ロジック"""
    if not logs:
        yield send_sse({"content": "分析対象データがありません。"})
        return

    summary = "\n".join(
        [
            f"- 評価:{l.get('rating','-')} | {l.get('comment','-')[:100]}"
            for l in logs[:50]
        ]
    )
    prompt = prompts.FEEDBACK_ANALYSIS.format(summary=summary)

    try:
        stream = await llm_service.generate_stream(prompt)
        async for chunk in stream:
            text = chunk.text if hasattr(chunk, "text") else str(chunk)
            if text:
                yield send_sse({"content": text})
    except Exception as e:
        logging.error(f"Feedback analysis error: {e}", exc_info=True)
        yield send_sse({"content": f"分析エラー: {e}"})