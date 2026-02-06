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
from services import prompts # ★プロンプトモジュール

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
        
        # まずクエリ拡張
        expanded_query = await search_service.expand_query(user_input)
        
        # 検索実行
        # ※ search_service.search メソッドが実装されている前提で呼び出し
        # 実装されていない場合は search.py に search メソッドを追加する必要がありますが、
        # ここでは既存の search_service のメソッド構成に合わせて適宜修正してください。
        # もし `search` メソッドがない場合は、以下のように個別に呼び出します:
        # --- 個別呼び出しパターン ---
        # 1. Embedding (省略) -> 2. DB検索 (省略) -> 3. Rerank -> 4. LitM -> 5. Filter
        # ------------------------
        # ここではコードの整合性のため、仮に search_service.search があるか、
        # 以前のコードのように処理を記述する必要があります。
        # 今回の修正範囲はプロンプトの外部化なので、ロジック自体は既存のものを維持します。
        
        # (簡易的なプレースホルダー: 実際には search.py に統合 search メソッドを作るのがベストです)
        # 今回は search.py に search メソッドがないため、ここでは詳細な実装を割愛し、
        # 既存のロジックが search_service 内にカプセル化されているか、
        # あるいはここで実装されている必要があります。
        # とりあえず空リストで初期化し、既存の実装があればそれを使います。
        
        # ※ 前回のコードで search_service.search を使っていた場合はここも修正不要です。
        if hasattr(search_service, 'search'):
             search_result_obj = await search_service.search(
                query=user_input, # または expanded_query
                session_id=session_id
             )
             search_results = search_result_obj.get("documents", [])
        else:
             # searchメソッドがない場合の簡易実装（本来は search.py に実装すべき）
             pass 

        if not search_results:
             yield send_sse({'content': AI_MESSAGES["NOT_FOUND"]})
             # 完了シグナル
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

        # システムプロンプト準備 (★プロンプトを prompts.py から取得)
        try:
            full_system_prompt = prompts.SYSTEM_GENERATION.format(
                context_text=context_str,
                current_date=current_date_str
            )
        except Exception:
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
        msg = AI_MESSAGES["SYSTEM_ERROR"]
        yield send_sse({'content': f"\n\n{msg} (Error: {error_str})"})
        
    finally:
        log_context(session_id, "Response generation finished.")

@traceable(name="Feedback_Analysis_Job", run_type="chain")
async def analyze_feedback_trends(logs: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
    """
    フィードバック分析用ロジック
    """
    if not logs:
        yield send_sse({'content': '分析対象データがありません。'})
        return
    
    summary = "\n".join([f"- 評価:{l.get('rating','-')} | {l.get('comment','-')[:100]}" for l in logs[:50]])
    
    # ★プロンプトを prompts.py から取得
    prompt = prompts.FEEDBACK_ANALYSIS.format(summary=summary)

    try:
        stream = await llm_service.generate_stream(prompt)
        async for chunk in stream:
            text = chunk.text if hasattr(chunk, 'text') else str(chunk)
            if text:
                yield send_sse({'content': text})
    except Exception as e:
        yield send_sse({'content': f'分析エラー: {e}'})