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
    "QA_SIMILARITY_THRESHOLD": 0.95, # FAQの即答ライン
    "RERANK_SCORE_THRESHOLD": 6.0,   # リランク足切りライン(0-10)
    "MAX_HISTORY_LENGTH": 20,
}

# セーフティ設定
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}

# ★修正点1: エラーメッセージの正確な定義
AI_MESSAGES = {
    "NOT_FOUND": (
        "申し訳ありません。ご質問に関連する確実な情報が資料内に見つかりませんでした。"
        "大学窓口へ直接お問い合わせいただくことをお勧めします。"
    ),
    "RATE_LIMIT": "申し訳ありません。現在アクセスが集中しています。1分ほど待ってから再度お試しください。",
    "SYSTEM_ERROR": "システムエラーが発生しました。しばらく時間をおいて再度お試しください。",
    "BLOCKED": "生成された回答がセーフティガイドラインに抵触したため、表示できませんでした。言い回しを変えて再度お試しください。"
}

# スレッドプール（CPUバウンドな処理用）
executor = ThreadPoolExecutor(max_workers=4)

# -----------------------------------------------------------------------------
# 2. プロンプト & スキーマ定義 (Structured Outputs用)
# -----------------------------------------------------------------------------

# リランク出力用の型定義
class RankedItem(typing.TypedDict):
    id: int
    score: float
    reason: str

class RerankResponse(typing.TypedDict):
    ranked_items: list[RankedItem]

# プロンプトテンプレート
PROMPT_RERANK = """
ユーザーの質問に対し、以下のドキュメントが回答根拠として適切か0-10点で採点してください。
質問: {query}
候補:
{candidates_text}
"""

PROMPT_SYSTEM_GENERATION = """
あなたは札幌学院大学の学生サポートAIです。
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
    """API制限(429)対策: エラーメッセージから待機時間を解析してリトライ"""
    max_retries = 3
    default_delay = 5
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
# 4. コアロジック: 検索パイプライン
# -----------------------------------------------------------------------------
class SearchPipeline:
    @staticmethod
    async def optimize_query(user_query: str, session_id: str) -> str:
        """HyDE + Query Expansion (必要に応じて有効化)"""
        # ※API節約のため、現在は使用していないが機能として残す
        prompt = f"""
        ユーザーの質問に基づいて、大学のデータベース検索に最適な「検索キーワード」を作成してください。
        専門用語への言い換えを含め、出力は検索用テキストのみにしてください。
        質問: "{user_query}"
        """
        try:
            model = genai.GenerativeModel(USE_MODEL)
            resp = await api_request_with_retry(
                model.generate_content_async, prompt, safety_settings=SAFETY_SETTINGS
            )
            return resp.text.strip()
        except Exception:
            return user_query

    @staticmethod
    async def rerank(query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """Gemini Structured Outputs を使用した高速・確実なリランク"""
        if not documents:
            return []
        
        # コンテキスト作成 (トークン節約のため、先頭1000文字程度に制限)
        candidates_text = ""
        for i, doc in enumerate(documents):
            meta = doc.get('metadata', {})
            snippet = doc.get('content', '')[:1000].replace('\n', ' ')
            candidates_text += f"ID:{i} [Source:{meta.get('source', '?')}]\n{snippet}\n\n"

        formatted_prompt = PROMPT_RERANK.format(query=query, candidates_text=candidates_text)

        try:
            model = genai.GenerativeModel(USE_MODEL)
            # ★改善: response_schemaで型安全にJSONを取得
            resp = await api_request_with_retry(
                model.generate_content_async,
                formatted_prompt,
                generation_config=GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=RerankResponse
                ),
                safety_settings=SAFETY_SETTINGS
            )
            
            # JSONパース処理
            data = json.loads(resp.text)
            
            reranked = []
            for item in data.get("ranked_items", []):
                idx = item.get("id")
                score = item.get("score")
                
                # インデックスの妥当性とスコアチェック
                if idx is not None and 0 <= idx < len(documents):
                    if score >= PARAMS["RERANK_SCORE_THRESHOLD"]:
                        doc = documents[idx]
                        doc['rerank_score'] = score
                        reranked.append(doc)
            
            reranked.sort(key=lambda x: x['rerank_score'], reverse=True)
            return reranked[:top_k]

        except Exception as e:
            logging.error(f"Rerank Error: {e}")
            # エラー時は元の順序の上位をそのまま返す（フェイルセーフ）
            return documents[:top_k]

    @staticmethod
    async def filter_diversity(documents: List[Dict], threshold: float = 0.7) -> List[Dict]:
        """MMR風フィルタリング（重複排除）"""
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

def _build_references(response_text: str, sources_map: Dict[int, str]) -> str:
    """回答生成後に参照元リンクを作成するヘルパー関数"""
    unique_refs = []
    seen_sources = set()
    
    for idx, src in sources_map.items():
        if src in seen_sources: continue
        # テキスト内で引用されているか、または上位3件なら表示
        if f"[{idx}]" in response_text or idx <= 3:
            unique_refs.append(f"* [{idx}] {src}")
            seen_sources.add(src)
            
    if unique_refs:
        return "\n\n## 参照元\n" + "\n".join(unique_refs)
    return ""

# -----------------------------------------------------------------------------
# 5. メイン: チャットロジック
# -----------------------------------------------------------------------------
async def enhanced_chat_logic(request: Request, chat_req: ChatQuery):
    session_id = get_or_create_session_id(request)
    feedback_id = str(uuid.uuid4())
    user_input = chat_req.query.strip()
    
    # クライアントへ初期レスポンス
    yield send_sse({'feedback_id': feedback_id, 'status_message': '🔍 データベースを検索しています...'})

    try:
        # Step 1: Embeddingの非同期実行
        embedding_task = asyncio.create_task(
            genai.embed_content_async(
                model=chat_req.embedding_model,
                content=user_input,
                task_type="retrieval_query"
            )
        )

        # Step 2: Embedding結果の取得
        try:
            raw_emb_result = await embedding_task
            query_embedding = raw_emb_result["embedding"]
        except Exception as e:
            log_context(session_id, f"Embedding Failed: {e}", "error")
            # ★修正点2: Embedding失敗は「システムエラー」として通知（アクセス集中ではない）
            yield send_sse({'content': AI_MESSAGES["SYSTEM_ERROR"]})
            return

        # Step 3: FAQ (QA Database) チェック
        # 高スコアでヒットすれば即return
        if qa_hits := core_database.db_client.search_fallback_qa(query_embedding, match_count=1):
            top_qa = qa_hits[0]
            if top_qa.get('similarity', 0) >= PARAMS["QA_SIMILARITY_THRESHOLD"]:
                resp = format_urls_as_links(f"よくあるご質問に回答が見つかりました。\n\n---\n{top_qa['content']}")
                history_manager.add(session_id, "assistant", resp)
                yield send_sse({'content': resp, 'show_feedback': True, 'feedback_id': feedback_id})
                return

        # Step 4: ドキュメント検索 (Hybrid)
        # 処理節約のためクエリ拡張はスキップし、生の入力を使用
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
        
        # Step 5: フィルタリング & リランク
        unique_docs = await SearchPipeline.filter_diversity(raw_docs)
        
        # Geminiによるリランク実行
        relevant_docs = await SearchPipeline.rerank(user_input, unique_docs[:15], top_k=chat_req.top_k)

        if not relevant_docs:
            yield send_sse({'content': AI_MESSAGES["NOT_FOUND"]})
            return

        # Step 6: 回答生成
        yield send_sse({'status_message': '✍️ 回答を執筆しています...'})
        
        context_parts = []
        sources_map = {} # {doc_id: source_name}
        
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
        
        # ★修正点3: 回答が空（セーフティ等でブロック）の場合のハンドリング追加
        if not full_resp:
             yield send_sse({'content': AI_MESSAGES["BLOCKED"]})
             history_manager.add(session_id, "assistant", "[[BLOCKED]]")
             return

        # Step 7: 参照元リンクの追記
        if "情報が見つかりません" not in full_resp:
            refs_text = _build_references(full_resp, sources_map)
            if refs_text:
                yield send_sse({'content': refs_text})
                full_resp += refs_text
        
        history_manager.add(session_id, "assistant", full_resp)

    except Exception as e:
        log_context(session_id, f"Critical Pipeline Error: {e}", "error")
        
        # ★修正点4: エラー種別によるメッセージの正確な出し分け
        error_str = str(e)
        if "429" in error_str or "Quota" in error_str:
            msg = AI_MESSAGES["RATE_LIMIT"]
        elif "finish_reason" in error_str: # Gemini固有のブロックエラーなど
            msg = AI_MESSAGES["BLOCKED"]
        else:
            msg = AI_MESSAGES["SYSTEM_ERROR"]
            
        yield send_sse({'content': msg})
        
    finally:
        # どのような終了フローでもフィードバックボタンは表示する
        yield send_sse({'show_feedback': True, 'feedback_id': feedback_id})

# -----------------------------------------------------------------------------
# 6. 分析機能 (管理者用)
# -----------------------------------------------------------------------------
async def analyze_feedback_trends(logs: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
    if not logs:
        yield send_sse({'content': '分析対象データがありません。'})
        return
    
    # ログデータの要約
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