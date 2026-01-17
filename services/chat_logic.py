import logging
import uuid
import json
import asyncio
import re
import os
from typing import List, Dict, Any, AsyncGenerator, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
import typing_extensions as typing

# 外部ライブラリ
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig
from fastapi import Request
from dotenv import load_dotenv

# 内部モジュール
from core.config import GEMINI_API_KEY
from core import database as core_database
from models.schemas import ChatQuery
from services.utils import format_urls_as_links

# -----------------------------------------------------------------------------
# 1. 設定 & 定数定義
# -----------------------------------------------------------------------------
load_dotenv()
genai.configure(api_key=GEMINI_API_KEY)

# 使用モデル
USE_MODEL = "gemini-2.5-flash"

# パラメータ
PARAMS = {
    "QA_SIMILARITY_THRESHOLD": 0.90, # FAQの即答ライン
    "RERANK_SCORE_THRESHOLD": 4.0,   # リランク足切りライン
    "MAX_HISTORY_LENGTH": 20,
}

# セーフティ設定
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}

# エラーメッセージ
AI_MESSAGES = {
    "NOT_FOUND": (
        "申し訳ありません。ご質問に関連する確実な情報が資料内に見つかりませんでした。"
        "大学窓口へ直接お問い合わせいただくことをお勧めします。"
    ),
    "RATE_LIMIT": "現在アクセスが集中しています。1分ほど待ってから再度お試しください。",
    "SYSTEM_ERROR": "システムエラーが発生しました。しばらく時間をおいて再度お試しください。",
    "BLOCKED": "生成された回答がガイドラインに抵触したため表示できませんでした。"
}

executor = ThreadPoolExecutor(max_workers=4)

# -----------------------------------------------------------------------------
# 2. プロンプト定義
# -----------------------------------------------------------------------------

class RankedItem(typing.TypedDict):
    id: int
    score: float
    reason: str

class RerankResponse(typing.TypedDict):
    ranked_items: list[RankedItem]

PROMPT_RERANK = """
あなたは検索システムの評価AIです。
ユーザーの質問に対し、以下のドキュメントが回答の根拠としてどれほど適切か、0点から10点で採点してください。
質問: {query}
候補ドキュメント:
{candidates_text}
"""

PROMPT_SYSTEM_GENERATION = """
あなたは**札幌学院大学の学生サポートAI**です。
提供された <context> タグ内の情報**のみ**を使用して、親しみやすく丁寧な言葉遣いで回答してください。

# 重要な回答ルール（厳守）
1. **情報源の限定**:
   - 必ず提供された <context> 内の情報に基づいて回答してください。
   - **<context> に記載がない事項については、自身の知識や一般常識で補完せず、必ず「資料内に情報が見つからない」旨を伝えてください。**
   - 推測や「一般的には〜」といった回答は禁止します。

2. **引用（インライン引用）**:
   - 回答の根拠となる事実の末尾に、必ず `[1]` や `[1][2]` の形式で資料IDを付記してください。
   - 文末だけでなく、重要な数値や条件のすぐ後ろに付けてください。

3. **回答のトーンと構成**:
   - 冒頭に「こんにちは！札幌学院大学の学生サポートAIです。」という挨拶と、共感的な一言を添えてください。
   - 専門用語や複雑な計算式は、太字、箇条書き、水平線（---）を活用し、視覚的にわかりやすく整理してください。
   - 例：計算式は水平線で挟むなどして強調してください。

4. **ハルシネーションの徹底防止**:
   - 大学名や制度名が <context> 内で特定できない場合は、断定を避けてください。
"""

# -----------------------------------------------------------------------------
# 3. ユーティリティ & クラス
# -----------------------------------------------------------------------------

def get_or_create_session_id(
    source: Union[str, Request, None] = None, 
    query_obj: Optional[ChatQuery] = None
) -> str:
    """
    セッションIDを取得または生成します。
    """
    # 1. 文字列が直接渡された場合
    if isinstance(source, str):
        return source

    # 2. ChatQueryにIDがある場合 (最優先)
    if query_obj and hasattr(query_obj, 'session_id') and query_obj.session_id:
        return query_obj.session_id
    
    # 3. Requestオブジェクトから取得
    if isinstance(source, Request):
        if hasattr(source, "session"):
            sid = source.session.get('chat_session_id')
            if not sid:
                sid = str(uuid.uuid4())
                source.session['chat_session_id'] = sid
            return sid

    # 4. 解決できない場合は新規発行
    return str(uuid.uuid4())

def log_context(session_id: str, message: str, level: str = "info"):
    msg = f"[Session: {session_id}] {message}"
    getattr(logging, level, logging.info)(msg)

def send_sse(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

async def api_request_with_retry(func, *args, **kwargs):
    max_retries = 3
    default_delay = 4
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "Quota" in error_str:
                if attempt == max_retries - 1:
                    logging.error(f"API Quota Exceeded: {e}")
                    raise e
                match = re.search(r"retry in (\d+\.?\d*)s", error_str)
                wait_time = float(match.group(1)) + 1.0 if match else default_delay * (2 ** attempt)
                logging.warning(f"Rate limit hit. Waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
            else:
                raise e

# --- HistoryManager ---
class ChatHistoryManager:
    def __init__(self):
        pass
    
    @property
    def supabase(self):
        """実際に必要になったタイミングでクライアントを取得"""
        if core_database.db_client is None or getattr(core_database.db_client, 'client', None) is None:
            logging.error("Database client is not initialized.")
            return None
        return core_database.db_client.client

    def add(self, session_id: str, role: str, content: str):
        if not self.supabase: return
        try:
            self.supabase.table("chat_history").insert({
                "session_id": session_id,
                "role": role,
                "content": content
            }).execute()
        except Exception as e:
            logging.error(f"History add failed: {e}")

    def get_context_string(self, session_id: str, limit: int = 10) -> str:
        if not self.supabase: return ""
        try:
            res = self.supabase.table("chat_history")\
                .select("role, content, created_at")\
                .eq("session_id", session_id)\
                .order("created_at", desc=True)\
                .limit(limit)\
                .execute()
            if not res.data: return ""
            history = sorted(res.data, key=lambda x: x['created_at'])
            return "\n".join([f"{h['role']}: {h['content']}" for h in history])
        except Exception as e:
            logging.error(f"History fetch failed: {e}")
            return ""

history_manager = ChatHistoryManager()

# -----------------------------------------------------------------------------
# 4. 検索パイプライン
# -----------------------------------------------------------------------------
class SearchPipeline:
    @staticmethod
    async def rerank(query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        if not documents: return []
        
        candidates_text = ""
        for i, doc in enumerate(documents):
            meta = doc.get('metadata', {})
            snippet = doc.get('content', '')[:300].replace('\n', ' ')
            candidates_text += f"ID:{i} [Source:{meta.get('source', '?')}]\n{snippet}\n\n"

        formatted_prompt = PROMPT_RERANK.format(query=query, candidates_text=candidates_text)

        try:
            model = genai.GenerativeModel(USE_MODEL)
            resp = await api_request_with_retry(
                model.generate_content_async,
                formatted_prompt,
                generation_config=GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=RerankResponse
                ),
                safety_settings=SAFETY_SETTINGS
            )
            data = json.loads(resp.text)
            reranked = []
            for item in data.get("ranked_items", []):
                idx = item.get("id")
                score = item.get("score")
                if idx is not None and 0 <= idx < len(documents):
                    if score >= PARAMS["RERANK_SCORE_THRESHOLD"]:
                        doc = documents[idx]
                        doc['rerank_score'] = score
                        reranked.append(doc)
            reranked.sort(key=lambda x: x['rerank_score'], reverse=True)
            return reranked[:top_k]
        except Exception as e:
            logging.error(f"Rerank Error: {e}")
            return documents[:top_k]

    @staticmethod
    async def filter_diversity(documents: List[Dict], threshold: float = 0.7) -> List[Dict]:
        loop = asyncio.get_running_loop()
        unique_docs = []
        def _calc_sim(a, b): return SequenceMatcher(None, a, b).ratio()

        for doc in documents:
            content = doc.get('content', '')
            is_duplicate = False
            for selected in unique_docs:
                sim = await loop.run_in_executor(executor, _calc_sim, content, selected.get('content', ''))
                if sim > threshold:
                    is_duplicate = True; break
            if not is_duplicate: unique_docs.append(doc)
        return unique_docs

    @staticmethod
    def reorder_documents(documents: List[Dict]) -> List[Dict]:
        if not documents: return []
        first_half = documents[0::2]
        second_half = documents[1::2][::-1]
        return first_half + second_half

# -----------------------------------------------------------------------------
# 5. 検索・参照ユーティリティ
# -----------------------------------------------------------------------------

def get_signed_url(file_path: str, bucket_name: str = "images"):
    """
    非公開ストレージ内のファイルに対して、1時間有効な署名付きURLを発行します。
    """
    try:
        # 非公開の 'images' バケットからアクセス権付きのURLを生成
        response = core_database.db_client.client.storage.from_(bucket_name).create_signed_url(file_path, 3600)
        
        if isinstance(response, dict) and "signedURL" in response:
            return response["signedURL"]
        return response 
    except Exception as e:
        logging.error(f"署名付きURLの発行に失敗しました (Path: {file_path}): {e}")
        return None

def _build_references(response_text: str, sources_map: Dict[int, str]) -> str:
    """
    回答内の [1] などの引用タグに基づき、クリック可能な画像リンクを生成します。
    """
    unique_refs = []
    seen_sources = set()
    # 本文中の [1] などの数字をすべて抽出
    cited_ids = set(map(int, re.findall(r'\[(\d+)\]', response_text)))
    
    for idx, src in sources_map.items():
        # 引用されたID、または最初の2件を常に表示
        if idx in cited_ids or idx <= 2:
            if src in seen_sources: continue
            
            # 非公開ストレージ対応：署名付きURLを取得
            signed_url = get_signed_url(src)
            
            if signed_url:
                # onclickイベントでJavaScriptにURLを渡す
                unique_refs.append(
                    f"* <a href='#' class='source-link' "
                    f"data-url='{signed_url}' "
                    f"onclick='event.preventDefault(); showSourceImage(this.dataset.url); return false;'>"
                    f"{src}</a>"
                )
            else:
                # URL取得失敗時のフォールバック
                unique_refs.append(f"* {src} (プレビュー不可)")
                
            seen_sources.add(src)
            
    if unique_refs:
        # 資料の「どこに書いてあったかを表示する」ルールに基づく 
        return "\n\n### 参照元データ\n" + "\n".join(unique_refs)
    return ""

# -----------------------------------------------------------------------------
# 6. メインチャットロジック
# -----------------------------------------------------------------------------
async def enhanced_chat_logic(request: Request, query_obj: ChatQuery):
    """
    【重要】引数の順序は (request, query_obj) です。
    """
    # セッションIDの取得
    session_id = get_or_create_session_id(request, query_obj)
    
    feedback_id = str(uuid.uuid4())
    user_input = query_obj.query.strip()
    
    yield send_sse({
        'feedback_id': feedback_id, 
        'status_message': '🔍 データベースを検索しています...',
        'type': 'status'
    })

    full_resp = ""

    try:
        # Step 1: Embedding
        embedding_task = asyncio.create_task(
            genai.embed_content_async(
                model=query_obj.embedding_model,
                content=user_input,
                task_type="retrieval_query"
            )
        )
        
        try:
            raw_emb_result = await embedding_task
            query_embedding = raw_emb_result["embedding"]
        except Exception as e:
            log_context(session_id, f"Embedding Failed: {e}", "error")
            yield send_sse({'content': AI_MESSAGES["SYSTEM_ERROR"]})
            return

        # Step 2: FAQ Check
        if core_database.db_client:
            qa_hits = core_database.db_client.search_fallback_qa(query_embedding, match_count=1)
            if qa_hits and qa_hits[0].get('similarity', 0) >= PARAMS["QA_SIMILARITY_THRESHOLD"]:
                top_qa = qa_hits[0]
                resp = format_urls_as_links(f"よくあるご質問に情報がありました。\n\n---\n{top_qa['content']}")
                history_manager.add(session_id, "assistant", resp)
                yield send_sse({'content': resp, 'show_feedback': True, 'feedback_id': feedback_id})
                return

            # Step 3: Search
            raw_docs = core_database.db_client.search_documents_hybrid(
                collection_name=query_obj.collection,
                query_text=user_input, 
                query_embedding=query_embedding,
                match_count=30
            )
        else:
            raw_docs = []

        if not raw_docs:
            yield send_sse({'content': AI_MESSAGES["NOT_FOUND"]})
            return

        yield send_sse({'status_message': '🧐 AIが文献を読んで選定中...', 'type': 'status'})

        # Step 4: Pipeline
        unique_docs = await SearchPipeline.filter_diversity(raw_docs)
        reranked_docs = await SearchPipeline.rerank(user_input, unique_docs[:15], top_k=query_obj.top_k)
        relevant_docs = SearchPipeline.reorder_documents(reranked_docs)

        if not relevant_docs:
            yield send_sse({'content': AI_MESSAGES["NOT_FOUND"]})
            return

        # Step 5: Generation
        yield send_sse({'status_message': '✍️ 回答を執筆しています...', 'type': 'status'})
        
        context_parts = []
        sources_map = {}
        
        for idx, doc in enumerate(relevant_docs, 1):
            src = doc.get('metadata', {}).get('source', '不明')
            sources_map[idx] = src
            context_parts.append(f"<doc id='{idx}' src='{src}'>\n{doc.get('content','')}\n</doc>")
        
        context_str = "\n".join(context_parts)
        history_str = history_manager.get_context_string(session_id)
        
        full_system_prompt = f"""{PROMPT_SYSTEM_GENERATION}
        
### 検索された資料
{context_str}

### これまでの会話
{history_str}
"""
        model = genai.GenerativeModel(USE_MODEL)
        stream = await api_request_with_retry(
            model.generate_content_async,
            f"ユーザーの質問: {user_input}",
            stream=True,
            generation_config=GenerationConfig(temperature=0.0), # 資料に忠実にする
            safety_settings=SAFETY_SETTINGS
        )
        
        yield send_sse({'status_message': '', 'type': 'status'})

        async for chunk in stream:
            if chunk.text:
                full_resp += chunk.text
                yield send_sse({'content': chunk.text})
        
        if not full_resp:
             yield send_sse({'content': AI_MESSAGES["BLOCKED"]})
             return

        # Step 6: References
        if "情報が見つかりません" not in full_resp:
            # ここで修正版の _build_references を呼び出します
            refs_text = _build_references(full_resp, sources_map)
            if refs_text:
                yield send_sse({'content': refs_text})
                full_resp += refs_text
        
        history_manager.add(session_id, "assistant", full_resp)

    except Exception as e:
        log_context(session_id, f"Critical Pipeline Error: {e}", "error")
        if not full_resp:
            yield send_sse({'content': AI_MESSAGES["SYSTEM_ERROR"]})
            
    finally:
        yield send_sse({'show_feedback': True, 'feedback_id': feedback_id})

# -----------------------------------------------------------------------------
# 7. 分析機能
# -----------------------------------------------------------------------------
async def analyze_feedback_trends(logs: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
    if not logs:
        yield send_sse({'content': 'データなし'})
        return
    summary = "\n".join([f"- {l.get('rating','-')} | {l.get('comment','-')[:50]}" for l in logs[:30]])
    try:
        model = genai.GenerativeModel(USE_MODEL)
        stream = await api_request_with_retry(
            model.generate_content_async, 
            f"分析と改善提案:\n{summary}", 
            stream=True
        )
        async for chunk in stream:
            if chunk.text: yield send_sse({'content': chunk.text})
    except Exception as e:
        yield send_sse({'content': str(e)})