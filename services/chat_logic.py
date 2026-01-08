import logging
import uuid
import json
import asyncio
import re
from typing import List, Dict, Any, AsyncGenerator, Optional
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from fastapi import Request

# 内部モジュール
from core.config import GEMINI_API_KEY
from core import database as core_database
from models.schemas import ChatQuery
from services.utils import format_urls_as_links

# -----------------------------------------------------------------------------
# 設定 & 定数
# -----------------------------------------------------------------------------
genai.configure(api_key=GEMINI_API_KEY)

# ★修正: 2026年現在の最新安定版を指定
USE_MODEL = "gemini-2.5-flash"

PARAMS = {
    "QA_SIMILARITY_THRESHOLD": 0.95,
    "RERANK_SCORE_THRESHOLD": 6.5,
    "MAX_HISTORY_LENGTH": 20,
}

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
    "ERROR": "現在アクセスが集中しており回答できません。しばらく時間をおいて再度お試しください。",
}

executor = ThreadPoolExecutor(max_workers=4)

# -----------------------------------------------------------------------------
# ユーティリティ関数
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

def clean_and_parse_json(text: str) -> Dict[str, Any]:
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'^```\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {}

async def api_request_with_retry(func, *args, **kwargs):
    """
    API制限(429)対策: エラーメッセージから待機時間を解析してリトライ
    """
    max_retries = 3
    default_delay = 5  # 解析できなかった場合のデフォルト待機時間
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_str = str(e)
            # エラーメッセージに 429 や Quota が含まれていたらリトライ処理へ
            if "429" in error_str or "Quota" in error_str:
                if attempt == max_retries - 1:
                    logging.error(f"API Quota Exceeded after {max_retries} retries.")
                    raise e
                # エラーメッセージから "retry in 55.2s" のような秒数を抽出
                wait_time = default_delay
                match = re.search(r"retry in (\d+\.?\d*)s", error_str)
                if match:
                    # 指示された秒数 + 1秒（念のため）待機
                    wait_time = float(match.group(1)) + 1.0
                else:
                    # 見つからない場合は指数バックオフ (5s, 10s...)
                    wait_time = default_delay * (2 ** attempt)

                logging.warning(f"Rate limit hit. Google requested wait: {wait_time:.1f}s. Retrying...")
                # ユーザーを待たせすぎないよう、ログには出すが処理は継続
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
# コアロジック: 検索パイプライン
# -----------------------------------------------------------------------------
class SearchPipeline:
    @staticmethod
    async def optimize_query(user_query: str, session_id: str) -> str:
        """HyDE + Query Expansion"""
        prompt = f"""
        ユーザーの質問に基づいて、大学のデータベース検索に最適な「検索キーワード」を作成してください。
        専門用語への言い換え（例: "取り消し" -> "履修中止"）を含め、出力は検索用テキストのみにしてください。
        ユーザーの質問: "{user_query}"
        """
        try:
            model = genai.GenerativeModel(USE_MODEL)
            # リトライ付きで呼び出し
            resp = await api_request_with_retry(
                model.generate_content_async, prompt, safety_settings=SAFETY_SETTINGS
            )
            optimized = resp.text.strip()
            log_context(session_id, f"クエリ拡張: {optimized}")
            return optimized
        except Exception:
            return user_query

    @staticmethod
    async def rerank(query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """検索結果のリランク処理"""
        if not documents:
            return []
        
        candidates_text = ""
        for i, doc in enumerate(documents):
            meta = doc.get('metadata', {})
            snippet = doc.get('content', '')[:2000].replace('\n', ' ')
            candidates_text += f"ID:{i} [Source:{meta.get('source', '?')}]\n{snippet}\n\n"

        prompt = f"""
        ユーザーの質問に対し、以下のドキュメントが回答根拠として適切か0-10点で採点してください。
        質問: {query}
        候補:
        {candidates_text}
        出力形式(JSON): {{ "ranked_items": [{{ "id": int, "score": float, "reason": str }}] }}
        """
        try:
            model = genai.GenerativeModel(USE_MODEL)
            resp = await api_request_with_retry(
                model.generate_content_async, prompt, safety_settings=SAFETY_SETTINGS
            )
            data = clean_and_parse_json(resp.text)
            reranked = []
            for item in data.get("ranked_items", []):
                idx, score = int(item.get("id", -1)), float(item.get("score", 0))
                if 0 <= idx < len(documents) and score >= PARAMS["RERANK_SCORE_THRESHOLD"]:
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

# -----------------------------------------------------------------------------
# メイン: チャットロジック
# -----------------------------------------------------------------------------
async def enhanced_chat_logic(request: Request, chat_req: ChatQuery):
    session_id = get_or_create_session_id(request)
    feedback_id = str(uuid.uuid4())
    user_input = chat_req.query.strip()
    yield send_sse({'feedback_id': feedback_id})

    try:
        # --- 1. 検索フェーズ ---
        yield send_sse({'status_message': '🔍 データベースを検索しています...'})
        # [削減ポイント1] クエリ拡張 (optimize_query) を廃止
        # task_query = asyncio.create_task(SearchPipeline.optimize_query(user_input, session_id))
        search_query = user_input  # ユーザーの入力をそのまま使う

        # [削減ポイント2] Embeddingを1回だけ実行し、FAQと文書検索の両方で使い回す
        # Note: ユーザー入力をベクトル化
        embedding_task = asyncio.create_task(
            genai.embed_content_async(
                model=chat_req.embedding_model,
                content=user_input,
                task_type="retrieval_query"
            )
        )

        # A. FAQチェック (埋め込み完了を待つ)
        try:
            # Embeddingタスクの結果取得
            raw_emb_result = await embedding_task
            query_embedding = raw_emb_result["embedding"]

            # A. FAQ (Q&A) チェック
            if qa_hits := core_database.db_client.search_fallback_qa(query_embedding, match_count=1):
                top_qa = qa_hits[0]
                if top_qa.get('similarity', 0) >= PARAMS["QA_SIMILARITY_THRESHOLD"]:
                    # FAQヒット時はここで終了。リランクも回答生成も走らないのでAPI消費は最小
                    resp = format_urls_as_links(f"よくあるご質問に回答が見つかりました。\n\n---\n{top_qa['content']}")
                    history_manager.add(session_id, "assistant", resp)
                    yield send_sse({'content': resp, 'show_feedback': True, 'feedback_id': feedback_id})
                    return
        except Exception as e:
            log_context(session_id, f"FAQ Search/Embed Error: {e}", "warning")
            # 万が一Embeddingに失敗していたら、この後の検索もできないためエラー終了
            if 'query_embedding' not in locals():
                yield send_sse({'content': AI_MESSAGES["ERROR"]})
                return

        # B. DB検索
        # クエリ拡張をしていないので、さきほど取得した query_embedding をそのまま流用 (再度のAPIコール不要)
        raw_docs = core_database.db_client.search_documents_hybrid(
            collection_name=chat_req.collection,
            query_text=search_query,       # 生の質問文
            query_embedding=query_embedding, # さっきのベクトル
            match_count=30                 # リランク前なので少し広めに取る
        )
        yield send_sse({'status_message': '🧐 AIが文献を読んで選定中...'})
        unique_docs = await SearchPipeline.filter_diversity(raw_docs)
        # ---------------------------------------------------------
        # [修正] リランクを実行（API制限時の救済措置付き）
        # ---------------------------------------------------------
        relevant_docs = []
        try:
            # APIが生きていれば、リランクを実行して精度を高める
            # 候補を15件渡し、上位 top_k 件に絞り込む
            relevant_docs = await SearchPipeline.rerank(user_input, unique_docs[:15], top_k=chat_req.top_k)
        except Exception as e:
            # ★ここが重要: API制限(429)などでエラーが出た場合の「命綱」
            log_context(session_id, f"Rerank API Failed (Fallback used): {e}", "warning")
            # エラー時は無理にリランクせず、DB検索のスコア順（上位5件）をそのまま使う
            # これにより、APIエラーが出ても回答不能にならず、最低限の結果を返せる
            relevant_docs = unique_docs[:5]

        # --- 2. 回答生成フェーズ ---
        if not relevant_docs:
            yield send_sse({'content': AI_MESSAGES["NOT_FOUND"]})
        else:
            yield send_sse({'status_message': '✍️ 回答を執筆しています...'})
            context_parts = []
            sources_map = {}
            for idx, doc in enumerate(relevant_docs, 1):
                src = doc.get('metadata', {}).get('source', '不明')
                sources_map[idx] = src
                context_parts.append(f"<doc id='{idx}' src='{src}'>\n{doc.get('content','')}\n</doc>")
            system_prompt = f"""
            あなたは札幌学院大学の学生サポートAIです。
            以下の<context>内の情報**のみ**を使用して、質問に回答してください。

            # 回答のルール
            1. **根拠の紐付け**:
            文章中の重要な事実には、文末に `[1]` のように**短い番号のみ**を付記してください。
            2. **形式**:
            - 学生に寄り添った、丁寧で親しみやすい「です・ます」調。
            - 読みやすいように箇条書きや**太字**を活用する。
            - 情報がない場合は「情報が見つかりません」と答える。
            <context>
            {chr(10).join(context_parts)}
            </context>
            """
            model = genai.GenerativeModel(USE_MODEL)
            # [2回目の生成APIコール] 回答生成
            stream = await api_request_with_retry(
                model.generate_content_async,
                [system_prompt, f"質問: {user_input}"],
                stream=True,
                safety_settings=SAFETY_SETTINGS
            )
            full_resp = ""
            async for chunk in stream:
                if chunk.text:
                    full_resp += chunk.text
                    yield send_sse({'content': chunk.text})
            # 参照元の追記
            if "情報が見つかりません" not in full_resp:
                refs_header = "\n\n## 参照元\n"
                unique_refs = []
                seen_sources = set()

                for idx, src in sources_map.items():
                    if src in seen_sources: continue
                    # 本文で参照されているか、上位3件までは表示
                    if f"[{idx}]" in full_resp or idx <= 3:
                        unique_refs.append(f"* [{idx}] {src}")
                        seen_sources.add(src)

                if unique_refs:
                    refs_text = refs_header + "\n".join(unique_refs)
                    yield send_sse({'content': refs_text})
                    full_resp += refs_text
            history_manager.add(session_id, "assistant", full_resp)

    except Exception as e:
        log_context(session_id, f"Critical Error: {e}", "error")
        if "429" in str(e) or "Quota" in str(e):
             yield send_sse({'content': "申し訳ありません。現在アクセスが集中しています。恐れ入りますが、1分ほど待ってから再度お試しください。"})
        else:
             yield send_sse({'content': AI_MESSAGES["ERROR"]})
    finally:
        yield send_sse({'show_feedback': True, 'feedback_id': feedback_id})

# -----------------------------------------------------------------------------
# 管理者用機能
# -----------------------------------------------------------------------------
async def analyze_feedback_trends(logs: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
    if not logs:
        yield send_sse({'content': '分析対象データがありません。'})
        return

    summary = "\n".join([f"- 評価:{l.get('rating','-')} | {l.get('comment','-')[:100]}" for l in logs[:50]])
    prompt = f"""
    チャットボット利用ログの分析レポートをMarkdownで作成してください。
    データ:
    {summary}
    項目: 1.ユーザートレンド, 2.低評価の原因, 3.改善案
    """
    try:
        model = genai.GenerativeModel(USE_MODEL)
        stream = await api_request_with_retry(model.generate_content_async, prompt, stream=True)
        async for chunk in stream:
            if chunk.text:
                yield send_sse({'content': chunk.text})
    except Exception as e:
        yield send_sse({'content': f'分析エラー: {e}'})