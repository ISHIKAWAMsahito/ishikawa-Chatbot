import logging
import uuid
import json
import asyncio
import re
import os
from typing import List, Dict, Any, AsyncGenerator, Optional
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
import typing_extensions as typing

# 外部ライブラリ
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig
from fastapi import Request
from dotenv import load_dotenv

# ★追加: Supabase関連
from supabase import create_client, Client

# 内部モジュール
from core.config import GEMINI_API_KEY
# ★追加: configからSupabaseのキーを読み込むと仮定
# (もしcore/config.pyにない場合は、os.getenv("SUPABASE_URL")などで直接取得してください)
from core.config import SUPABASE_URL, SUPABASE_SERVICE_KEY 
from core import database as core_database
from models.schemas import ChatQuery
from services.utils import format_urls_as_links

# -----------------------------------------------------------------------------
# 1. 設定 & 定数定義
# -----------------------------------------------------------------------------
load_dotenv()
genai.configure(api_key=GEMINI_API_KEY)

# ★追加: Supabaseクライアントの初期化
# バケット名（非公開テーブルに相当）
STORAGE_BUCKET_NAME = "images" 
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# 使用モデル
USE_MODEL = "gemini-2.5-flash"

# パラメータ
PARAMS = {
    "QA_SIMILARITY_THRESHOLD": 0.90,
    "RERANK_SCORE_THRESHOLD": 6.0,
    "MAX_HISTORY_LENGTH": 20,
}

# セーフティ設定
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}

AI_MESSAGES = {
    "NOT_FOUND": (
        "申し訳ありません。ご質問に関連する確実な情報が資料内に見つかりませんでした。"
        "大学窓口へ直接お問い合わせいただくことをお勧めします。"
    ),
    "RATE_LIMIT": "申し訳ありません。現在アクセスが集中しています。1分ほど待ってから再度お試しください。",
    "SYSTEM_ERROR": "システムエラーが発生しました。しばらく時間をおいて再度お試しください。",
    "BLOCKED": "生成された回答がセーフティガイドラインに抵触したため、表示できませんでした。言い回しを変えて再度お試しください。"
}

# スレッドプール
executor = ThreadPoolExecutor(max_workers=4)

# -----------------------------------------------------------------------------
# 2. プロンプト & スキーマ定義
# -----------------------------------------------------------------------------
class RankedItem(typing.TypedDict):
    id: int
    score: float
    reason: str

class RerankResponse(typing.TypedDict):
    ranked_items: list[RankedItem]

PROMPT_RERANK = """
ユーザーの質問に対し、以下のドキュメントが回答根拠として適切か0-10点で採点してください。
質問: {query}
候補:
{candidates_text}
"""

PROMPT_SYSTEM_GENERATION = """
あなたは札幌学院大学のサポートAIです。
以下の<context>内の情報**のみ**を使用して、質問に回答してください。

# 回答のルール
1. **根拠の紐付け**:
   文章中の重要な事実には、文末に `[1]` のように**短い番号のみ**を付記してください。
2. **形式**:
   - 学生に寄り添った、丁寧で親しみやすい「です・ます」調。
   - 読みやすいように箇条書きや**太字**を活用する。
   - 情報がない場合は「情報が見つかりません」と答える。
"""

# -----------------------------------------------------------------------------
# 3. ユーティリティ関数
# -----------------------------------------------------------------------------
def get_or_create_session_id(request: Request) -> str:
    session_id = request.session.get('chat_session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session['chat_session_id'] = session_id
    return session_id

def log_context(session_id: str, message: str, level: str = "info"):
    msg = f"[Session: {session_id}] {message}"
    getattr(logging, level, logging.info)(msg)

def send_sse(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

async def api_request_with_retry(func, *args, **kwargs):
    max_retries = 3
    default_delay = 5
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "Quota" in error_str:
                if attempt == max_retries - 1:
                    raise e
                wait_time = default_delay * (2 ** attempt)
                match = re.search(r"retry in (\d+\.?\d*)s", error_str)
                if match:
                    wait_time = float(match.group(1)) + 1.0
                await asyncio.sleep(wait_time)
            else:
                raise e

class ChatHistoryManager:
    def __init__(self):
        self._histories: Dict[str, List[Dict[str, str]]] = {}

    def add(self, session_id: str, role: str, content: str):
        if session_id not in self._histories:
            self._histories[session_id] = []
        self._histories[session_id].append({"role": role, "content": content})
        if len(self._histories[session_id]) > PARAMS["MAX_HISTORY_LENGTH"]:
            self._histories[session_id] = self._histories[session_id][-PARAMS["MAX_HISTORY_LENGTH"]:]

history_manager = ChatHistoryManager()

# -----------------------------------------------------------------------------
# 4. コアロジック: 検索パイプライン & ★参照リンク生成
# -----------------------------------------------------------------------------

# ★追加: Supabase署名付きURL生成ロジック
def _generate_signed_url_sync(filename: str) -> Optional[str]:
    """
    Supabase Storageから署名付きURLを取得する（同期関数）。
    txtファイルの場合は、同名の画像ファイル(png, jpg)の存在もチェックしてURL化を試みる。
    """
    try:
        # 1. そのままのファイル名でトライ
        # create_signed_url returns Dict with 'signedURL' key usually
        res = supabase.storage.from_(STORAGE_BUCKET_NAME).create_signed_url(filename, 3600)
        if res and 'signedURL' in res:
            return res['signedURL']
    except Exception:
        pass

    # 2. 拡張子が .txt の場合、画像ファイル (.png, .jpg) があるか試行する
    # （テキストデータが画像からOCRされたものである場合への対応）
    if filename.endswith(".txt"):
        base_name = os.path.splitext(filename)[0]
        for ext in [".png", ".jpg", ".jpeg", ".pdf"]:
            try:
                image_filename = f"{base_name}{ext}"
                res = supabase.storage.from_(STORAGE_BUCKET_NAME).create_signed_url(image_filename, 3600)
                if res and 'signedURL' in res:
                    return res['signedURL']
            except Exception:
                continue
    
    return None

async def _build_references_async(response_text: str, sources_map: Dict[int, str]) -> str:
    """
    回答生成後に参照元リンクを作成する（非同期並列処理版）。
    Supabaseへのアクセスを並列化して高速化を図る。
    """
    unique_refs = []
    seen_sources = set()
    
    # 処理対象のリストアップ
    target_items = []
    for idx, src in sources_map.items():
        if src in seen_sources: continue
        # テキスト内で引用されているか、または上位3件なら表示対象
        if f"[{idx}]" in response_text or idx <= 3:
            target_items.append((idx, src))
            seen_sources.add(src)
    
    if not target_items:
        return ""

    # スレッドプールで並列にURL発行
    loop = asyncio.get_running_loop()
    tasks = []
    for _, src in target_items:
        tasks.append(loop.run_in_executor(executor, _generate_signed_url_sync, src))
    
    # 全てのURL取得を待機
    signed_urls = await asyncio.gather(*tasks)
    
    # 結果の整形
    for (idx, src), url in zip(target_items, signed_urls):
        if url:
            # 署名付きURLが取得できた場合: リンク化
            # ファイル名が見やすいように basename のみを表示しても良いが、ここでは識別のためsrc全体を表示
            display_name = os.path.basename(src)
            unique_refs.append(f"* [{idx}] [{display_name}]({url}) ⏳リンク有効期限:1時間")
        else:
            # 取得失敗した場合: 通常のテキスト表示
            unique_refs.append(f"* [{idx}] {src}")

    if unique_refs:
        return "\n\n## 参照元 (クリックで資料を表示)\n" + "\n".join(unique_refs)
    return ""

class SearchPipeline:
    @staticmethod
    async def rerank(query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
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
        
        def _calc_sim(a, b):
            return SequenceMatcher(None, a, b).ratio()

        for doc in documents:
            content = doc.get('content', '')
            is_duplicate = False
            for selected in unique_docs:
                sim = await loop.run_in_executor(executor, _calc_sim, content, selected.get('content', ''))
                if sim > threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_docs.append(doc)
        return unique_docs

# -----------------------------------------------------------------------------
# 5. メイン: チャットロジック
# -----------------------------------------------------------------------------
async def enhanced_chat_logic(request: Request, chat_req: ChatQuery):
    session_id = get_or_create_session_id(request)
    feedback_id = str(uuid.uuid4())
    user_input = chat_req.query.strip()
    
    yield send_sse({'feedback_id': feedback_id, 'status_message': '🔍 データベースを検索しています...'})

    try:
        embedding_task = asyncio.create_task(
            genai.embed_content_async(
                model=chat_req.embedding_model,
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

        # FAQチェック
        if qa_hits := core_database.db_client.search_fallback_qa(query_embedding, match_count=1):
            top_qa = qa_hits[0]
            if top_qa.get('similarity', 0) >= PARAMS["QA_SIMILARITY_THRESHOLD"]:
                resp = format_urls_as_links(f"よくあるご質問に回答が見つかりました。\n\n---\n{top_qa['content']}")
                history_manager.add(session_id, "assistant", resp)
                yield send_sse({'content': resp, 'show_feedback': True, 'feedback_id': feedback_id})
                return

        # ドキュメント検索
        raw_docs = core_database.db_client.search_documents_hybrid(
            collection_name=chat_req.collection,
            query_text=user_input, 
            query_embedding=query_embedding,
            match_count=30
        )

        if not raw_docs:
            yield send_sse({'content': AI_MESSAGES["NOT_FOUND"]})
            return

        yield send_sse({'status_message': '🧐 AIが文献を読んで選定中...'})
        
        unique_docs = await SearchPipeline.filter_diversity(raw_docs)
        relevant_docs = await SearchPipeline.rerank(user_input, unique_docs[:15], top_k=chat_req.top_k)

        if not relevant_docs:
            yield send_sse({'content': AI_MESSAGES["NOT_FOUND"]})
            return

        yield send_sse({'status_message': '✍️ 回答を執筆しています...'})
        
        context_parts = []
        sources_map = {} 
        
        for idx, doc in enumerate(relevant_docs, 1):
            src = doc.get('metadata', {}).get('source', '不明')
            sources_map[idx] = src
            context_parts.append(f"<doc id='{idx}' src='{src}'>\n{doc.get('content','')}\n</doc>")
        
        context_str = "\n".join(context_parts)
        full_system_prompt = f"{PROMPT_SYSTEM_GENERATION}\n<context>\n{context_str}\n</context>"

        model = genai.GenerativeModel(USE_MODEL)
        stream = await api_request_with_retry(
            model.generate_content_async,
            [full_system_prompt, f"質問: {user_input}"],
            stream=True,
            safety_settings=SAFETY_SETTINGS
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

        # ★修正: 非同期で参照元リンク（Signed URL）を生成して追記
        if "情報が見つかりません" not in full_resp:
            yield send_sse({'status_message': '🔗 参照リンクを生成中...'})
            refs_text = await _build_references_async(full_resp, sources_map)
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

# -----------------------------------------------------------------------------
# 6. 分析機能
# -----------------------------------------------------------------------------
async def analyze_feedback_trends(logs: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
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
        model = genai.GenerativeModel(USE_MODEL)
        stream = await api_request_with_retry(model.generate_content_async, prompt, stream=True)
        async for chunk in stream:
            if chunk.text:
                yield send_sse({'content': chunk.text})
    except Exception as e:
        yield send_sse({'content': f'分析エラー: {e}'})