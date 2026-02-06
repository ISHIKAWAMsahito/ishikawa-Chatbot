# services/chat_logic.py
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
# utilsから新しい関数 format_references をインポート
from services.utils import (
    get_or_create_session_id, 
    send_sse, 
    log_context, 
    ChatHistoryManager, 
    format_urls_as_links,
    format_references 
)
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
    user_input = chat_req.question # models.schemas.ChatQueryのフィールド名に合わせてください(question or query)
    
    # LangSmith用のRunTree取得（エラーハンドリング用）
    run_tree = get_current_run_tree()

    log_context(session_id, f"Start processing query: {user_input}")

    # 日時取得（JST）
    JST = timezone(timedelta(hours=9), 'JST')
    now = datetime.now(JST)
    current_date_str = now.strftime("%Y年%m月%d日")

    # 検索結果を保持する変数
    search_results = []
    
    try:
        # 1. 履歴の追加
        history_manager.add(session_id, "user", user_input)
        
        yield send_sse({'status_message': '🔍 質問を分析しています...'})

        # 2. 検索 (Search Service)
        # SearchServiceの実装に合わせて呼び出しを調整してください
        # ここではハイブリッド検索を行い、documentsリストが返ってくると仮定します
        
        # まずクエリ拡張
        expanded_query = await search_service.expand_query(user_input)
        
        # 検索実行（内部でEmbedding化、DB検索、リランクなどを行う想定）
        # ※ search_service.search メソッドが存在し、必要な処理をラップしている場合
        search_result_obj = await search_service.search(
            query=user_input,
            session_id=session_id
        )
        search_results = search_result_obj.get("documents", [])
        
        # もし search_service.search がない場合は、元のコードのようにステップごとに記述します：
        if not search_results:
             # Embedding生成など（省略：元のロジックが必要ならここに戻す）
             # 簡易的な実装例として search_service に委譲しています
             pass

        if not search_results:
             yield send_sse({'content': AI_MESSAGES["NOT_FOUND"]})
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
        except KeyError:
             full_system_prompt = prompts.SYSTEM_GENERATION.format(context_text=context_str)

        # ストリーミング回答の開始
        ai_response_full = ""
        
        async for chunk in llm_service.generate_response_stream(
            query=user_input,
            context_docs=search_results, # 互換性のため渡す
            history=chat_history,
            system_prompt=full_system_prompt # 生成したプロンプトを渡す
        ):
            text_chunk = chunk if isinstance(chunk, str) else chunk.get("content", "")
            ai_response_full += text_chunk
            yield send_sse({'content': text_chunk})

        # 4. 参照元リストの生成と送信 (★修正ポイント)
        # metadataにurlがある場合はリンク化された参照リストが生成される
        references_text = format_references(search_results)
        
        if references_text:
            # AIの回答の後に改行を入れて参照元を追記送信
            yield send_sse({'content': references_text})
            # ログ保存用に全文にも結合しておく
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
        msg = AI_MESSAGES["SYSTEM_ERROR"]
        yield send_sse({'content': f"\n\n{msg} (Error: {error_str})"})
        
    finally:
        log_context(session_id, "Response generation finished.")