import logging
import uuid
import json
import asyncio
import re
from typing import List, Dict, Any, AsyncGenerator, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold
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

# チューニングパラメータ
PARAMS = {
    "STRICT_THRESHOLD": 0.80,
    "QA_SIMILARITY_THRESHOLD": 0.95,
    "RERANK_SCORE_THRESHOLD": 6.5,
    "MAX_HISTORY_LENGTH": 20,
    "MAX_CONTEXT_CHAR_LENGTH": 60000,
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
    "ERROR": "システムエラーが発生しました。管理者に連絡してください。",
}

# CPU処理用スレッドプール
executor = ThreadPoolExecutor(max_workers=4)

# -----------------------------------------------------------------------------
# ユーティリティ関数
# -----------------------------------------------------------------------------
def get_or_create_session_id(request: Request) -> str:
    """セッションIDの管理"""
    session_id = request.session.get('chat_session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session['chat_session_id'] = session_id
    return session_id

def log_context(session_id: str, message: str, level: str = "info"):
    """構造化ログ出力"""
    msg = f"[Session: {session_id}] {message}"
    getattr(logging, level, logging.info)(msg)

def send_sse(data: Dict[str, Any]) -> str:
    """SSE形式のレスポンス作成ヘルパー"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

def clean_and_parse_json(text: str) -> Dict[str, Any]:
    """Geminiの出力を安全にJSONパース"""
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'^```\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {}

class ChatHistoryManager:
    """簡易メモリ内履歴管理"""
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
    """検索・リランク・フィルタリングを一元管理するクラス"""

    @staticmethod
    async def optimize_query(user_query: str, session_id: str) -> str:
        """HyDE + Query Expansion"""
        prompt = f"""
        ユーザーの質問に基づいて、大学のデータベース検索に最適な「検索キーワード」を作成してください。
        専門用語への言い換え（例: "取り消し" -> "履修中止"）を含め、出力は検索用テキストのみにしてください。
        ユーザーの質問: "{user_query}"
        """
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            resp = await model.generate_content_async(prompt, safety_settings=SAFETY_SETTINGS)
            optimized = resp.text.strip()
            log_context(session_id, f"クエリ拡張: {optimized}")
            return optimized
        except Exception:
            return user_query

    @staticmethod
    async def check_ambiguity(query: str) -> Dict[str, Any]:
        """意図の曖昧性判定"""
        # ルールベースの高速判定
        if len(query) > 10 and any(x in query for x in ["方法", "場所", "申請", "について", "教え"]):
            return {"is_ambiguous": False}

        prompt = f"""
        あなたはヘルプデスクAIです。以下の質問が回答に十分な具体性を持っているか判定してください。
        質問: "{query}"
        出力形式(JSON): {{ "is_ambiguous": bool, "response_text": str, "candidates": [str] }}
        - 単語のみ等の場合は true とし、誘導尋問を response_text に記述。
        - candidates には想定される具体的な質問例を列挙。
        """
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            resp = await model.generate_content_async(prompt, safety_settings=SAFETY_SETTINGS)
            return clean_and_parse_json(resp.text)
        except Exception:
            return {"is_ambiguous": False}

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
            model = genai.GenerativeModel("gemini-2.5-flash")
            resp = await model.generate_content_async(prompt, safety_settings=SAFETY_SETTINGS)
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
    """
    チャットボットのメイン処理フロー
    1. 意図理解 -> 2. 検索(FAQ/DB) -> 3. リランク -> 4. 回答生成
    """
    session_id = get_or_create_session_id(request)
    feedback_id = str(uuid.uuid4())
    user_input = chat_req.query.strip()
    # フロントエンド初期化用
    yield send_sse({'feedback_id': feedback_id})

    try:
        # --- 1. 意図理解フェーズ ---
        yield send_sse({'status_message': '🤔 質問の意図を分析しています...'})
        ambiguity = await SearchPipeline.check_ambiguity(user_input)
        if ambiguity.get("is_ambiguous"):
            resp = ambiguity.get("response_text", "もう少し具体的に教えていただけますか？")
            if candidates := ambiguity.get("candidates"):
                resp += "\n\n**もしかして:**\n" + "\n".join([f"- {c}" for c in candidates])
            yield send_sse({'content': resp, 'show_feedback': True, 'feedback_id': feedback_id})
            return

        # --- 2. 検索フェーズ (FAQ & DB) ---
        yield send_sse({'status_message': '🔍 データベースを検索しています...'})
        # クエリ拡張とEmbeddingを並列実行
        task_query = asyncio.create_task(SearchPipeline.optimize_query(user_input, session_id))
        task_embed = asyncio.create_task(
            genai.embed_content_async(model=chat_req.embedding_model, content=user_input, task_type="retrieval_query")
        )

        # A. FAQチェック
        try:
            raw_emb = (await task_embed)["embedding"]
            if qa_hits := core_database.db_client.search_fallback_qa(raw_emb, match_count=1):
                top_qa = qa_hits[0]
                if top_qa.get('similarity', 0) >= PARAMS["QA_SIMILARITY_THRESHOLD"]:
                    task_query.cancel() # DB検索不要
                    resp = format_urls_as_links(f"よくあるご質問に回答が見つかりました。\n\n---\n{top_qa['content']}")
                    history_manager.add(session_id, "assistant", resp)
                    yield send_sse({'content': resp, 'show_feedback': True, 'feedback_id': feedback_id})
                    return
        except Exception as e:
            log_context(session_id, f"FAQ Search Skip: {e}", "warning")

        # B. DB検索
        search_query = await task_query
        optimized_emb = (await genai.embed_content_async(
            model=chat_req.embedding_model, content=search_query, task_type="retrieval_query"
        ))["embedding"]

        raw_docs = core_database.db_client.search_documents_hybrid(
            collection_name=chat_req.collection,
            query_text=search_query,
            query_embedding=optimized_emb,
            match_count=30
        )
        # 多様性フィルタとリランク
        yield send_sse({'status_message': '🧐 文献の重要度をAIが精査中...'})
        unique_docs = await SearchPipeline.filter_diversity(raw_docs)
        relevant_docs = await SearchPipeline.rerank(user_input, unique_docs[:12], top_k=chat_req.top_k)

        # --- 3. 回答生成フェーズ ---
        if not relevant_docs:
            yield send_sse({'content': AI_MESSAGES["NOT_FOUND"]})
        else:
            yield send_sse({'status_message': '✍️ 回答を執筆しています...'})
            # コンテキスト構築
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
            model = genai.GenerativeModel(chat_req.model)
            stream = await model.generate_content_async(
                [system_prompt, f"質問: {user_input}"], stream=True, safety_settings=SAFETY_SETTINGS
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
                    if src in seen_sources:
                        continue
                    # 文中で引用された([n])、または検索スコア上位3件のみを表示
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
        yield send_sse({'content': AI_MESSAGES["ERROR"]})
    finally:
        yield send_sse({'show_feedback': True, 'feedback_id': feedback_id})

# -----------------------------------------------------------------------------
# 管理者用機能
# -----------------------------------------------------------------------------
async def analyze_feedback_trends(logs: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
    """フィードバック分析（管理者ダッシュボード用）"""
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
        model = genai.GenerativeModel("gemini-2.5-flash")
        async for chunk in await model.generate_content_async(prompt, stream=True):
            if chunk.text:
                yield send_sse({'content': chunk.text})
    except Exception as e:
        yield send_sse({'content': f'分析エラー: {e}'})