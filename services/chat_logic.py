# services/chat_logic.py
import logging
import uuid
from datetime import datetime, timedelta, timezone  # 日時操作用に追加
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
    format_urls_as_links
)
from services import prompts

# DI（依存性の注入）の準備
llm_service = LLMService()
search_service = SearchService(llm_service)
storage_service = StorageService()
history_manager = ChatHistoryManager(max_length=PARAMS["MAX_HISTORY_LENGTH"])

@traceable(name="Chat_Pipeline_Parent", run_type="chain")
async def enhanced_chat_logic(request: Request, chat_req: ChatQuery):
    """
    リファクタリング後のメインチャットロジック
    
    主な変更点:
    - 日本時間 (JST) での現在日時取得を追加
    - システムプロンプトへ現在日時 (current_date) を注入する処理を追加
    """
    session_id = get_or_create_session_id(request)
    user_input = chat_req.query.strip()
    feedback_id = str(uuid.uuid4())
    
    # ---------------------------------------------------------
    # 日時取得ロジック (JST固定)
    # ---------------------------------------------------------
    # サーバーのタイムゾーン設定に依存せず、常に日本時間を取得します。
    # これにより「現在が2025年度か」などをAIが正確に判断できます。
    JST = timezone(timedelta(hours=9), 'JST')
    now = datetime.now(JST)
    current_date_str = now.strftime("%Y年%m月%d日") # 例: 2025年10月27日
    
    # LangSmith: メタデータ（セッションIDなど）を追加
    run_tree = get_current_run_tree()
    if run_tree:
        run_tree.add_metadata({
            "session_id": session_id, 
            "user_query": user_input,
            "current_date_jst": current_date_str
        })

    yield send_sse({'feedback_id': feedback_id, 'status_message': '🔍 質問を分析しています...'})

    try:
        # -----------------------------------------------------
        # 1. クエリ拡張
        # -----------------------------------------------------
        expanded_query = await search_service.expand_query(user_input)
        
        # -----------------------------------------------------
        # 2. Embedding生成
        # -----------------------------------------------------
        query_embedding = await llm_service.get_embedding(
            text=expanded_query, 
            model=chat_req.embedding_model
        )

        # -----------------------------------------------------
        # 3. FAQチェック (Fallback)
        # -----------------------------------------------------
        if qa_hits := core_database.db_client.search_fallback_qa(query_embedding, match_count=1):
            top_qa = qa_hits[0]
            if top_qa.get('similarity', 0) >= PARAMS["QA_SIMILARITY_THRESHOLD"]:
                # FAQ回答にもリンク化処理を適用
                resp_content = top_qa['content']
                resp_formatted = format_urls_as_links(resp_content)
                
                formatted_response = f"よくあるご質問に回答が見つかりました。\n\n---\n{resp_formatted}"
                history_manager.add(session_id, "assistant", formatted_response)
                yield send_sse({'content': formatted_response, 'show_feedback': True, 'feedback_id': feedback_id})
                return

        yield send_sse({'status_message': '📚 資料を広く集めています...'})

        # -----------------------------------------------------
        # 4. DB検索 (Hybrid Search)
        # -----------------------------------------------------
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

        # -----------------------------------------------------
        # 5. フィルタリング & リランク & 並べ替え
        # -----------------------------------------------------
        unique_docs = search_service.filter_diversity(raw_docs)
        rerank_input = unique_docs[:PARAMS["RERANK_TOP_K_INPUT"]]
        
        # リランク処理
        # (注: prompts.py の RERANK プロンプトも最新情報の優先指示があることが望ましい)
        relevant_docs = await search_service.rerank(
            query=user_input, 
            documents=rerank_input, 
            top_k=chat_req.top_k
        )

        if not relevant_docs:
            yield send_sse({'content': AI_MESSAGES["NOT_FOUND"]})
            return

        # Lost in the Middle 対策 (重要なドキュメントを先頭と末尾に配置)
        final_docs = search_service.reorder_litm(relevant_docs)

        yield send_sse({'status_message': '✍️ 回答を生成しています...'})

        # -----------------------------------------------------
        # 6. コンテキスト構築
        # -----------------------------------------------------
        context_parts = []
        sources_map = {}
        for idx, doc in enumerate(final_docs, 1):
            meta = doc.get('metadata', {})
            src_display = meta.get('source', '不明')
            src_storage = meta.get('image_path', src_display)
            
            # AIへのヒントとして Source URL を明示
            # これによりファイル名に含まれる日付や年度情報もAIが認識しやすくなります
            doc_context = f"<doc id='{idx}' src='{src_display}'>\n"
            doc_context += f"Source Reference: {src_display}\n" 
            doc_context += f"Content: {doc.get('content','')}\n"
            doc_context += "</doc>"
            
            sources_map[idx] = {'display': src_display, 'storage': src_storage}
            context_parts.append(doc_context)
        
        context_str = "\n".join(context_parts)
        
        # -----------------------------------------------------
        # 7. 回答生成 (LLM)
        # -----------------------------------------------------
        # プロンプト内の {context_text} と {current_date} を埋め込む
        # 注: prompts.SYSTEM_GENERATION に {current_date} プレースホルダーが必要です
        try:
            full_system_prompt = prompts.SYSTEM_GENERATION.format(
                context_text=context_str,
                current_date=current_date_str
            )
        except KeyError:
            # 万が一 prompts.py が更新されていない場合のフォールバック
            logging.warning("SYSTEM_GENERATION prompt does not have 'current_date' placeholder.")
            full_system_prompt = prompts.SYSTEM_GENERATION.format(context_text=context_str)
        
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

        # -----------------------------------------------------
        # 8. 参照リンク生成と最終整形
        # -----------------------------------------------------
        final_content_updates = ""
        
        if "情報が見つかりません" not in full_resp:
            yield send_sse({'status_message': '🔗 参照リンクを生成中...'})
            
            # StorageServiceによる署名付きURL等の処理
            refs_text = await storage_service.build_references_async(full_resp, sources_map)
            
            if refs_text:
                full_resp += refs_text
                final_content_updates += refs_text

        # 最後に全文に対してURLリンク化処理を適用して履歴に保存
        formatted_full_resp = format_urls_as_links(full_resp)
        history_manager.add(session_id, "assistant", formatted_full_resp)

        # もし `refs_text` があった場合、それをクライアントに追送
        if final_content_updates:
            yield send_sse({'content': final_content_updates})

    except Exception as e:
        log_context(session_id, f"Critical Pipeline Error: {e}", "error")
        if run_tree:
            run_tree.end(error=str(e))
            
        error_str = str(e)
        if "429" in error_str or "Quota" in error_str:
            msg = AI_MESSAGES["RATE_LIMIT"]
        elif "finish_reason" in error_str:
            msg = AI_MESSAGES["BLOCKED"]
        else:
            msg = AI_MESSAGES["SYSTEM_ERROR"]
        yield send_sse({'content': msg})
        
    finally:
        # 完了またはエラー時にフィードバックUIを表示
        yield send_sse({'show_feedback': True, 'feedback_id': feedback_id})

@traceable(name="Feedback_Analysis_Job", run_type="chain")
async def analyze_feedback_trends(logs: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
    """
    フィードバック分析用ロジック (変更なし)
    """
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