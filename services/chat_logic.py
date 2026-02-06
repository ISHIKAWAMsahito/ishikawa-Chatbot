import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, AsyncGenerator
from fastapi import Request

# LangSmith トレース用
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

# 依存モジュールのインポート
from core import database as core_database
from core.constants import PARAMS, AI_MESSAGES
from models.schemas import ChatQuery
from services.llm import LLMService
from services.search import SearchService
from services.storage import StorageService
from services.utils import (
    get_or_create_session_id, 
    send_sse, 
    log_context, 
    ChatHistoryManager, 
    format_references 
)
# ★修正: プロンプトモジュールをインポート
from services import prompts

# DI（依存性の注入）の準備
llm_service = LLMService()
search_service = SearchService(llm_service)
storage_service = StorageService()
history_manager = ChatHistoryManager(max_length=PARAMS["MAX_HISTORY_LENGTH"])

@traceable(name="Chat_Pipeline_Parent", run_type="chain")
async def enhanced_chat_logic(request: Request, chat_req: ChatQuery) -> AsyncGenerator[str, None]:
    """
    RAGチャットロジック
    """
    session_id = get_or_create_session_id(request)
    user_input = chat_req.question or chat_req.query
    
    collection_name = getattr(chat_req, "collection", "student-knowledge-base")
    top_k = getattr(chat_req, "top_k", 5)
    embedding_model = getattr(chat_req, "embedding_model", "models/gemini-embedding-001")

    # LangSmith用のRunTree取得（エラーハンドリング用）
    run_tree = get_current_run_tree()

    log_context(session_id, f"Start processing query: {user_input}")

    # 日時取得（JST）
    JST = timezone(timedelta(hours=9), 'JST')
    now = datetime.now(JST)
    current_date_str = now.strftime("%Y年%m月%d日")

    search_results = []
    
    try:
        # 1. 履歴の追加
        history_manager.add(session_id, "user", user_input)
        
        yield send_sse({'status_message': '🔍 質問を分析しています...'})

        # 2. 検索 (Search Service)
        
        # まずクエリ拡張
        expanded_query = await search_service.expand_query(user_input)
        
        # 検索パイプライン実行 (Search -> Rerank -> LitM -> Filter)
        search_result_obj = await search_service.search(
            query=expanded_query, 
            session_id=session_id,
            collection_name=collection_name,
            top_k=top_k,
            embedding_model=embedding_model
        )
        search_results = search_result_obj.get("documents", [])
        
        # ヒットしなかった場合、元のクエリで再試行 (安全策)
        if not search_results and expanded_query != user_input:
             search_result_obj = await search_service.search(
                query=user_input,
                session_id=session_id,
                collection_name=collection_name,
                top_k=top_k,
                embedding_model=embedding_model
             )
             search_results = search_result_obj.get("documents", [])

        if not search_results:
             yield send_sse({'content': AI_MESSAGES.get("NOT_FOUND", "申し訳ありません。関連する情報が見つかりませんでした。")})
             # 履歴に保存して終了
             history_manager.add(session_id, "assistant", "関連情報が見つかりませんでした。")
             yield send_sse({'done': True, 'feedback_id': str(uuid.uuid4())})
             return

        yield send_sse({'status_message': '✍️ 回答を生成しています...'})

        # 3. 回答生成 (LLM Service)
        chat_history = history_manager.get_history(session_id)
        
        # コンテキスト構築
        context_parts = []
        for idx, doc in enumerate(search_results, 1):
            doc_content = doc.get('content', '')
            context_parts.append(f"<doc id='{idx}'>{doc_content}</doc>")
        context_str = "\n".join(context_parts)

        # システムプロンプト準備
        try:
            full_system_prompt = prompts.SYSTEM_GENERATION.format(
                context_text=context_str,
                current_date=current_date_str
            )
        except Exception:
             # フォーマットエラー等のフォールバック
             full_system_prompt = f"以下の情報を元に回答してください。\n{context_str}"

        # ストリーミング回答の開始
        ai_response_full = ""
        
        async for chunk in llm_service.generate_response_stream(
            query=user_input,
            context_docs=search_results, 
            history=chat_history,
            system_prompt=full_system_prompt
        ):
            text_chunk = chunk if isinstance(chunk, str) else chunk.get("content", "")
            ai_response_full += text_chunk
            yield send_sse({'content': text_chunk})

        # 4. 参照元リストの生成と送信
        references_text = format_references(search_results)
        
        if references_text:
            yield send_sse({'content': references_text})
            ai_response_full += references_text

        # 5. 履歴にAIの回答を保存
        history_manager.add(session_id, "assistant", ai_response_full)

        # 6. 完了シグナル
        feedback_id = str(uuid.uuid4())
        yield send_sse({'done': True, 'feedback_id': feedback_id})

    except Exception as e:
        log_context(session_id, f"Critical Pipeline Error: {e}", "error")
        if run_tree:
            run_tree.end(error=str(e))
            
        error_str = str(e)
        msg = AI_MESSAGES.get("SYSTEM_ERROR", "システムエラーが発生しました。")
        yield send_sse({'content': f"\n\n{msg} (Error: {error_str})"})
        
    finally:
        log_context(session_id, "Response generation finished.")

# ★この関数が不足していたためエラーになっていました。必ずファイルの末尾に含まれるようにしてください。
@traceable(name="Feedback_Analysis_Job", run_type="chain")
async def analyze_feedback_trends(logs: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
    """
    フィードバック分析用ロジック
    """
    if not logs:
        yield send_sse({'content': '分析対象データがありません。'})
        return
    
    summary = "\n".join([f"- 評価:{l.get('rating','-')} | {l.get('comment','-')[:100]}" for l in logs[:50]])
    
    # プロンプトモジュールから取得
    prompt = prompts.FEEDBACK_ANALYSIS.format(summary=summary)

    try:
        stream = await llm_service.generate_stream(prompt)
        async for chunk in stream:
            text = chunk.text if hasattr(chunk, 'text') else str(chunk)
            if text:
                yield send_sse({'content': text})
    except Exception as e:
        yield send_sse({'content': f'分析エラー: {e}'})