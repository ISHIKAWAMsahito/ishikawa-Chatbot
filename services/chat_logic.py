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

# パラメータ設定 (バランス調整済み)
PARAMS = {
    "QA_SIMILARITY_THRESHOLD": 0.90,  # DB内FAQの即答ライン
    "RERANK_SCORE_THRESHOLD": 4.0,    # リランク足切りライン (0-10)
    "DIVERSITY_THRESHOLD": 0.7,       # 重複排除の類似度ライン
    "MAX_HISTORY_LENGTH": 20,
}

# セーフティ設定
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}

# エラーメッセージ定義
AI_MESSAGES = {
    "NOT_FOUND": (
        "申し訳ありません。ご質問に関連する情報がデータベース（資料）内に見つかりませんでした。"
        "不確かな回答を避けるため、ここではお答えを控えさせていただきます。"
        "\n\n大学窓口へ直接お問い合わせいただくことをお勧めします。"
    ),
    "RATE_LIMIT": "申し訳ありません。現在アクセスが集中しています。1分ほど待ってから再度お試しください。",
    "SYSTEM_ERROR": "システムエラーが発生しました。しばらく時間をおいて再度お試しください。",
    "BLOCKED": "生成された回答がセーフティガイドラインに抵触したため、表示できませんでした。"
}

# スレッドプール（CPUバウンドな処理用）
executor = ThreadPoolExecutor(max_workers=4)

# -----------------------------------------------------------------------------
# 2. プロンプト定義 & スキーマ (Structured Outputs用)
# -----------------------------------------------------------------------------

# リランク出力用の型定義
class RankedItem(typing.TypedDict):
    id: int
    score: float
    reason: str

class RerankResponse(typing.TypedDict):
    ranked_items: list[RankedItem]

# プロンプトテンプレート (リランク用)
PROMPT_RERANK = """
あなたは厳格な査読者です。
ユーザーの質問に対し、以下のドキュメントが「回答の根拠」として使用できるかを0-10点で採点してください。

評価基準:
- 10点: 質問に対する直接的な答えが含まれている。
- 5-9点: 関連情報が含まれており、回答の構成に役立つ。
- 0-4点: キーワードは似ているが、文脈が異なる、または無関係。

質問: {query}
候補:
{candidates_text}
"""

PROMPT_SYSTEM_GENERATION = """
あなたは**札幌学院大学の学生サポートAI**です。
以下の <context> タグ内の情報**のみ**を使用して、質問に回答してください。

# 厳守すべきルール（ガードレール）

1. **情報の限定（Zero-Inference）**:
   - あなたが元々持っている知識（一般常識や他大学の事例）は一切使用しないでください。
   - **禁止事項**: 「一般的には」「通常は」「一般論として」といった表現は絶対に使用しないでください。
   - 文脈に答えが見つからない場合は、正直に「提供された資料内には、その情報が見当たりませんでした」と答えてください。

2. **引用フォーマットの徹底**:
   - 回答の根拠となる部分には、必ず `[1]` や `[1][2]` という形式で番号を振ってください。
   - **注意**: `(1)` や `Source: 1` は不可です。必ず `[` と `]` で囲んでください。（システムがリンクを生成するために必須です）

3. **トーンとマナー**:
   - 学生に寄り添った、親しみやすい「です・ます」調で話してください。
   - 冒頭は「こんにちは！札幌学院大学の学生サポートAIです。」で始めてください。
   - 専門用語や条件分岐が多い場合は、箇条書きや太字を使って視覚的に整理してください。

4. **回答プロセス**:
   - まず資料を読み、質問に関連する部分があるか確認する。
   - 一般論を混ぜないよう、資料にある事実だけを抽出して回答を構成する。
   - 引用番号 `[x]` が正しい位置にあるか確認してから出力する。
"""

# -----------------------------------------------------------------------------
# 3. ユーティリティ関数 & クラス
# -----------------------------------------------------------------------------

def get_or_create_session_id(
    source: Union[str, Request, None] = None, 
    query_obj: Optional[ChatQuery] = None
) -> str:
    """セッションIDを取得または生成します。"""
    if isinstance(source, str):
        return source
    if query_obj and hasattr(query_obj, 'session_id') and query_obj.session_id:
        return query_obj.session_id
    if isinstance(source, Request):
        if hasattr(source, "session"):
            sid = source.session.get('chat_session_id')
            if not sid:
                sid = str(uuid.uuid4())
                source.session['chat_session_id'] = sid
            return sid
    return str(uuid.uuid4())

def log_context(session_id: str, message: str, level: str = "info"):
    msg = f"[Session: {session_id}] {message}"
    getattr(logging, level, logging.info)(msg)

def send_sse(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

async def api_request_with_retry(func, *args, **kwargs):
    """API制限(429)対策: リトライロジック"""
    max_retries = 3
    default_delay = 4
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "Quota" in error_str:
                if attempt == max_retries - 1:
                    logging.error(f"API Quota Exceeded after {max_retries} retries.")
                    raise e
                
                wait_time = default_delay
                match = re.search(r"retry in (\d+\.?\d*)s", error_str)
                if match:
                    wait_time = float(match.group(1)) + 1.0
                else:
                    wait_time = default_delay * (2 ** attempt)

                logging.warning(f"Rate limit hit. Waiting {wait_time:.1f}s. Retrying...")
                await asyncio.sleep(wait_time)
            else:
                raise e

# --- HistoryManager ---
class ChatHistoryManager:
    def __init__(self):
        self._histories: Dict[str, List[Dict[str, str]]] = {}

    @property
    def supabase(self):
        if core_database.db_client is None or getattr(core_database.db_client, 'client', None) is None:
            return None
        return core_database.db_client.client

    def add(self, session_id: str, role: str, content: str):
        if self.supabase:
            try:
                self.supabase.table("chat_history").insert({
                    "session_id": session_id,
                    "role": role,
                    "content": content
                }).execute()
            except Exception as e:
                logging.error(f"History add failed: {e}")
        
        if session_id not in self._histories:
            self._histories[session_id] = []
        self._histories[session_id].append({"role": role, "content": content})
        if len(self._histories[session_id]) > PARAMS["MAX_HISTORY_LENGTH"]:
            self._histories[session_id] = self._histories[session_id][-PARAMS["MAX_HISTORY_LENGTH"]:]

    def get_context_string(self, session_id: str, limit: int = 10) -> str:
        if self.supabase:
            try:
                res = self.supabase.table("chat_history")\
                    .select("role, content, created_at")\
                    .eq("session_id", session_id)\
                    .order("created_at", desc=True)\
                    .limit(limit)\
                    .execute()
                if res.data:
                    history = sorted(res.data, key=lambda x: x['created_at'])
                    return "\n".join([f"{h['role']}: {h['content']}" for h in history])
            except Exception as e:
                logging.error(f"History fetch failed: {e}")
        
        hist = self._histories.get(session_id, [])[-limit:]
        return "\n".join([f"{h['role']}: {h['content']}" for h in hist])

history_manager = ChatHistoryManager()

# -----------------------------------------------------------------------------
# 4. 検索パイプライン
# -----------------------------------------------------------------------------
class SearchPipeline:
    @staticmethod
    async def rerank(query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """Gemini Structured Outputs を使用した高速・確実なリランク"""
        if not documents:
            return []
        
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
# 5. 参照リンク生成ユーティリティ (Supabase対応)
# -----------------------------------------------------------------------------

def get_signed_url(file_path: str, bucket_name: str = "images"):
    """
    非公開ストレージ内のファイルに対して、1時間有効な署名付きURLを発行します。
    """
    try:
        if core_database.db_client is None:
            logging.error("db_client is not initialized")
            return None

        # ファイル名に含まれる余分な空白を除去
        clean_path = file_path.strip()

        # 非公開の 'images' バケットからアクセス権付きのURLを生成(1時間有効)
        response = core_database.db_client.client.storage.from_(bucket_name).create_signed_url(clean_path, 3600)
        
        if isinstance(response, dict) and "signedURL" in response:
            return response["signedURL"]
        return response 
    except Exception as e:
        logging.error(f"Failed to get signed URL for {file_path}: {e}")
        return None

def _build_references(response_text: str, sources_map: Dict[int, Any]) -> str:
    """
    参照元のリンクを生成します。
    sources_mapの形式: {idx: {'source': str, 'metadata': dict}} または {idx: str} (後方互換性)
    """
    unique_refs = []
    seen_sources = set()
    cited_ids = set(map(int, re.findall(r'\[(\d+)\]', response_text)))
    
    for idx, source_info in sources_map.items():
        # 後方互換性: 文字列の場合
        if isinstance(source_info, str):
            src = source_info
            metadata = {}
        else:
            src = source_info.get('source', '不明')
            metadata = source_info.get('metadata', {})
        
        # 引用されている、または上位2つ以内の場合に表示
        if idx in cited_ids or idx <= 2:
            if src in seen_sources:
                continue
            
            # メタデータからURL情報を取得
            url = metadata.get('url')
            source_display = src
            
            # URLが存在する場合（Webスクレイピングなど）
            if url:
                # URLを直接リンクとして生成
                unique_refs.append(
                    f"* <a href='{url}' target='_blank' class='source-link' rel='noopener noreferrer'>"
                    f"{source_display}</a>"
                )
            else:
                # 画像ファイル用の署名付きURLを試す
                signed_url = get_signed_url(src)
                if signed_url:
                    unique_refs.append(
                        f"* <a href='#' class='source-link' data-url='{signed_url}' "
                        f"onclick='event.preventDefault(); showSourceImage(this.dataset.url); return false;'>"
                        f"{source_display}</a>"
                    )
                else:
                    # リンクがない場合はテキストのみ
                    unique_refs.append(f"* {source_display}")
            
            seen_sources.add(src)
            
    if unique_refs:
        return "\n\n### 参照元データ\n" + "\n".join(unique_refs)
    return ""

# -----------------------------------------------------------------------------
# 6. メインチャットロジック
# -----------------------------------------------------------------------------
async def enhanced_chat_logic(request: Request, query_obj: ChatQuery):
    session_id = get_or_create_session_id(request, query_obj)
    feedback_id = str(uuid.uuid4())
    user_input = query_obj.query.strip()
    full_resp = ""
    
    yield send_sse({
        'feedback_id': feedback_id, 
        'status_message': '🔍 データベースを検索しています...',
        'type': 'status'
    })

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

        # Step 2: Supabase QA (FAQ) Check
        if core_database.db_client:
            qa_hits = core_database.db_client.search_fallback_qa(query_embedding, match_count=1)
            if qa_hits and qa_hits[0].get('similarity', 0) >= PARAMS["QA_SIMILARITY_THRESHOLD"]:
                top_qa = qa_hits[0]
                resp = format_urls_as_links(f"よくあるご質問に情報がありました。\n\n---\n{top_qa['content']}")
                history_manager.add(session_id, "assistant", resp)
                yield send_sse({'content': resp, 'show_feedback': True, 'feedback_id': feedback_id})
                return

            # Step 3: Supabase Document Search (Hybrid)
            # 【調整】30件取得（エンジニア視点での最適化）
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

        yield send_sse({'status_message': '🧐 文献の重複を除去し、精査中...', 'type': 'status'})

        # Step 4: Pipeline (Filter -> Rerank -> Reorder)
        # 4-1. 重複排除 (MMR)
        unique_docs = await SearchPipeline.filter_diversity(raw_docs, threshold=PARAMS["DIVERSITY_THRESHOLD"])
        
        # 4-2. リランク (Geminiによるスコアリング)
        # 【調整】上位15件をリランク（レイテンシと精度のバランス重視）
        reranked_docs = await SearchPipeline.rerank(user_input, unique_docs[:15], top_k=query_obj.top_k)
        
        # 4-3. 再配置
        relevant_docs = SearchPipeline.reorder_documents(reranked_docs)

        if not relevant_docs:
            yield send_sse({'content': AI_MESSAGES["NOT_FOUND"]})
            return

        # Step 5: Generation
        yield send_sse({'status_message': '✍️ 回答を執筆しています...', 'type': 'status'})
        
        context_parts = []
        sources_map = {}
        for idx, doc in enumerate(relevant_docs, 1):
            metadata = doc.get('metadata', {})
            src = metadata.get('source', '不明')
            # sources_mapにメタデータ全体を含めて、URL情報なども参照できるようにする
            sources_map[idx] = {
                'source': src,
                'metadata': metadata
            }
            context_parts.append(f"<doc id='{idx}' src='{src}'>\n{doc.get('content','')}\n</doc>")
        
        context_str = "\n".join(context_parts)
        history_str = history_manager.get_context_string(session_id)
        
        full_system_prompt = f"""{PROMPT_SYSTEM_GENERATION}
        
### 検索された資料 (Supabase)
{context_str}

### これまでの会話
{history_str}
"""
        model = genai.GenerativeModel(USE_MODEL)
        stream = await api_request_with_retry(
            model.generate_content_async,
            f"ユーザーの質問: {user_input}",
            stream=True,
            generation_config=GenerationConfig(temperature=0.0), # 事実性重視
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
            refs_text = _build_references(full_resp, sources_map)
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
            
        if not full_resp:
            yield send_sse({'content': msg})
            
    finally:
        yield send_sse({'show_feedback': True, 'feedback_id': feedback_id})

# -----------------------------------------------------------------------------
# 7. 分析機能 (管理者用)
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