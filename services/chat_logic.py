import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, AsyncGenerator
from fastapi import Request
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

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
from services import prompts

llm_service = LLMService()
search_service = SearchService(llm_service)
storage_service = StorageService()
history_manager = ChatHistoryManager(max_length=PARAMS["MAX_HISTORY_LENGTH"])

@traceable(name="Chat_Pipeline_Parent", run_type="chain")
async def enhanced_chat_logic(request: Request, chat_req: ChatQuery) -> AsyncGenerator[str, None]:
    session_id = get_or_create_session_id(request)
    user_input = chat_req.question or chat_req.query
    
    collection_name = getattr(chat_req, "collection", "student-knowledge-base")
    top_k = getattr(chat_req, "top_k", 5)
    embedding_model = getattr(chat_req, "embedding_model", "models/gemini-embedding-001")

    run_tree = get_current_run_tree()
    log_context(session_id, f"Start processing query: {user_input}")

    JST = timezone(timedelta(hours=9), 'JST')
    current_date_str = datetime.now(JST).strftime("%Y年%m月%d日")

    search_results = []
    
    try:
        history_manager.add(session_id, "user", user_input)
        yield send_sse({'status_message': '🔍 質問を分析しています...'})

        # 1. クエリ拡張
        expanded_query = await search_service.expand_query(user_input)
        
        # 2. 検索パイプライン実行 (Search -> Rerank -> LitM -> Filter)
        search_result_obj = await search_service.search(
            query=expanded_query, 
            session_id=session_id,
            collection_name=collection_name,
            top_k=top_k,
            embedding_model=embedding_model
        )
        search_results = search_result_obj.get("documents", [])
        
        # ヒットなしの場合、元のクエリで再試行 (安全策)
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
             yield send_sse({'content': AI_MESSAGES.get("NOT_FOUND", "情報が見つかりませんでした。")})
             history_manager.add(session_id, "assistant", "情報なし")
             yield send_sse({'done': True, 'feedback_id': str(uuid.uuid4())})
             return

        yield send_sse({'status_message': '✍️ 回答を生成しています...'})

        # 3. 回答生成
        chat_history = history_manager.get_history(session_id)
        
        context_parts = []
        for idx, doc in enumerate(search_results, 1):
            doc_content = doc.get('content', '')
            context_parts.append(f"<doc id='{idx}'>{doc_content}</doc>")
        context_str = "\n".join(context_parts)

        try:
            full_system_prompt = prompts.SYSTEM_GENERATION.format(
                context_text=context_str,
                current_date=current_date_str
            )
        except Exception:
             full_system_prompt = f"以下の情報を元に回答してください。\n{context_str}"

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

        # 4. 参照元
        references_text = format_references(search_results)
        if references_text:
            yield send_sse({'content': references_text})
            ai_response_full += references_text

        history_manager.add(session_id, "assistant", ai_response_full)
        yield send_sse({'done': True, 'feedback_id': str(uuid.uuid4())})

    except Exception as e:
        log_context(session_id, f"Error: {e}", "error")
        if run_tree: run_tree.end(error=str(e))
        yield send_sse({'content': f"エラーが発生しました: {str(e)}"})