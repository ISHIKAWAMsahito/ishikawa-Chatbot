# services/chat_logic.py
import logging
import uuid
from typing import List, Dict, Any, AsyncGenerator
from fastapi import Request

# 依存モジュールのインポート
from core import database as core_database
from core.constants import PARAMS, AI_MESSAGES
from models.schemas import ChatQuery
from services.llm import LLMService
from services.search import SearchService
from services.storage import StorageService
from services.utils import get_or_create_session_id, send_sse, log_context, ChatHistoryManager
from services import prompts
from services.utils import format_urls_as_links

# DI（依存性の注入）の準備
llm_service = LLMService()
search_service = SearchService(llm_service)
storage_service = StorageService()
history_manager = ChatHistoryManager(max_length=PARAMS["MAX_HISTORY_LENGTH"])

async def enhanced_chat_logic(request: Request, chat_req: ChatQuery):
    """リファクタリング後のメインチャットロジック"""
    session_id = get_or_create_session_id(request)
    user_input = chat_req.query.strip()
    feedback_id = str(uuid.uuid4())
    
    yield send_sse({'feedback_id': feedback_id, 'status_message': '🔍 質問を分析しています...'})

    try:
        # 1. クエリ拡張
        expanded_query = await search_service.expand_query(user_input)
        
        # 2. Embedding生成 (★ここを修正: モデル名を渡す)
        query_embedding = await llm_service.get_embedding(
            text=expanded_query, 
            model=chat_req.embedding_model  # settings.pyの設定値がここに来ます
        )

        # 3. FAQチェック
        if qa_hits := core_database.db_client.search_fallback_qa(query_embedding, match_count=1):
            top_qa = qa_hits[0]
            if top_qa.get('similarity', 0) >= PARAMS["QA_SIMILARITY_THRESHOLD"]:
                resp = format_urls_as_links(f"よくあるご質問に回答が見つかりました。\n\n---\n{top_qa['content']}")
                history_manager.add(session_id, "assistant", resp)
                yield send_sse({'content': resp, 'show_feedback': True, 'feedback_id': feedback_id})
                return

        yield send_sse({'status_message': '📚 資料を広く集めています...'})

        # 4. DB検索 (Hybrid)
        raw_docs = core_database.db_client.search_documents_hybrid(
            collection_name=chat_req.collection,
            query_text=expanded_query,
            query_embedding=query_embedding,
            match_count=50
        )

        if not raw_docs:
            yield send_sse({'content': AI_MESSAGES["NOT_FOUND"]})
            return

        yield send_sse({'status_message': '🧐 AIが文献を精読・選別中...'})

        # 5. フィルタリング & リランク & 並べ替え
        unique_docs = search_service.filter_diversity(raw_docs)
        rerank_input = unique_docs[:PARAMS["RERANK_TOP_K_INPUT"]]
        
        relevant_docs = await search_service.rerank(
            query=user_input, 
            documents=rerank_input, 
            top_k=chat_req.top_k
        )

        if not relevant_docs:
            yield send_sse({'content': AI_MESSAGES["NOT_FOUND"]})
            return

        # Lost in the Middle 対策
        final_docs = search_service.reorder_litm(relevant_docs)

        yield send_sse({'status_message': '✍️ 回答を生成しています...'})

        # 6. コンテキスト構築
        context_parts = []
        sources_map = {}
        for idx, doc in enumerate(final_docs, 1):
            meta = doc.get('metadata', {})
            src_display = meta.get('source', '不明')
            src_storage = meta.get('image_path', src_display)
            
            sources_map[idx] = {'display': src_display, 'storage': src_storage}
            context_parts.append(f"<doc id='{idx}' src='{src_display}'>\n{doc.get('content','')}\n</doc>")
        
        context_str = "\n".join(context_parts)
        
        # 7. 回答生成
        full_system_prompt = f"{prompts.SYSTEM_GENERATION}\n<context>\n{context_str}\n</context>"
        
        stream = await llm_service.generate_stream(
            prompt=f"質問: {user_input}",
            system_prompt=full_system_prompt
        )
        
        full_resp = ""
        async for chunk in stream:
            if chunk.text:
                full_resp += chunk.text
                yield send_sse({'content': chunk.text})

        if not full_resp:
             yield send_sse({'content': AI_MESSAGES["BLOCKED"]})
             history_manager.add(session_id, "assistant", "[[BLOCKED]]")
             return

        # 8. 参照リンク生成
        if "情報が見つかりません" not in full_resp:
            yield send_sse({'status_message': '🔗 参照リンクを生成中...'})
            refs_text = await storage_service.build_references_async(full_resp, sources_map)
            if refs_text:
                yield send_sse({'content': refs_text})
                full_resp += refs_text
        
        history_manager.add(session_id, "assistant", full_resp)

    except Exception as e:
        log_context(session_id, f"Critical Pipeline Error: {e}", "error")
        error_str = str(e)
        if "429" in error_str or "Quota" in error_str:
            msg = AI_MESSAGES["RATE_LIMIT"]
        elif "finish_reason" in error_str:
            msg = AI_MESSAGES["BLOCKED"]
        else:
            msg = AI_MESSAGES["SYSTEM_ERROR"]
        yield send_sse({'content': msg})
        
    finally:
        yield send_sse({'show_feedback': True, 'feedback_id': feedback_id})

async def analyze_feedback_trends(logs: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
    """フィードバック分析用 (前回追加したものと同じ)"""
    if not logs:
        yield send_sse({'content': '分析対象データがありません。'})
        return
    
    summary = "\n".join([f"- 評価:{l.get('rating','-')} | {l.get('comment','-')[:100]}" for l in logs[:50]])
    prompt = f"""
    以下のチャットボット利用ログを分析し、Markdownでレポートを作成してください。
    # ログデータ
    {summary}
    # 出力項目
    1. ユーザーの主な関心事（トレンド）
    2. 低評価の原因と改善策
    3. 次のアクションプラン
    """
    try:
        stream = await llm_service.generate_stream(prompt)
        async for chunk in stream:
            if chunk.text:
                yield send_sse({'content': chunk.text})
    except Exception as e:
        yield send_sse({'content': f'分析エラー: {e}'})