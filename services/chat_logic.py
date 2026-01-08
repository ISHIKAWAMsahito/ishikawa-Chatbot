import logging
import uuid
import json
import asyncio
import re
from typing import List, Dict, Any, AsyncGenerator, Optional
import typing_extensions as typing
from concurrent.futures import ThreadPoolExecutor

from fastapi import Request
import google.generativeai as genai
from google.generativeai.types import (
    GenerationConfig,
    HarmCategory,
    HarmBlockThreshold
)
from difflib import SequenceMatcher

# -----------------------------------------------
# 外部モジュール・設定のインポート
# -----------------------------------------------
from core.config import GEMINI_API_KEY
from core import database as core_database
from models.schemas import ChatQuery
from services.utils import format_urls_as_links

# APIキー設定
genai.configure(api_key=GEMINI_API_KEY)

# -----------------------------------------------
# 定数・設定値
# -----------------------------------------------
STRICT_THRESHOLD = 0.80
QA_SIMILARITY_THRESHOLD = 0.95
RERANK_SCORE_THRESHOLD = 6.5    # 品質の担保のため少し厳しめに設定
MAX_HISTORY_LENGTH = 20

# 上限を少し絞り、精度低下（Lost in the Middle）を防ぐ
MAX_CONTEXT_CHAR_LENGTH = 60000

AI_NOT_FOUND_TOKEN = "[[NO_RELEVANT_INFO_FOUND]]"
AI_NOT_FOUND_MESSAGE_USER = (
    "申し訳ありません。ご質問に関連する確実な情報が資料内に見つかりませんでした。"
    "不正確な回答を避けるため、大学窓口へ直接お問い合わせいただくことをお勧めします。"
)

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}

# CPUバウンドな処理（類似度計算など）用のExecutor
executor = ThreadPoolExecutor(max_workers=4)

# -----------------------------------------------
# 構造化出力用の型定義
# -----------------------------------------------
class AmbiguityAnalysis(typing.TypedDict):
    is_ambiguous: bool
    response_text: str
    candidates: List[str]

class RerankItem(typing.TypedDict):
    id: int
    score: float
    reason: str

class RerankResult(typing.TypedDict):
    ranked_items: List[RerankItem]

# -----------------------------------------------
# ユーティリティ & ヘルパー
# -----------------------------------------------
def log_context(session_id: str, message: str, level: str = "info"):
    msg = f"[Session: {session_id}] {message}"
    if level == "error": logging.error(msg)
    elif level == "warning": logging.warning(msg)
    else: logging.info(msg)

class ChatHistoryManager:
    def __init__(self):
        self._histories: Dict[str, List[Dict[str, str]]] = {}

    def add_to_history(self, session_id: str, role: str, content: str):
        if session_id not in self._histories:
            self._histories[session_id] = []
        history = self._histories[session_id]
        history.append({"role": role, "content": content})
        if len(history) > MAX_HISTORY_LENGTH:
            self._histories[session_id] = history[-MAX_HISTORY_LENGTH:]

history_manager = ChatHistoryManager()

def get_or_create_session_id(request: Request) -> str:
    session_id = request.session.get('chat_session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session['chat_session_id'] = session_id
    return session_id

def _compute_similarity(text1: str, text2: str) -> float:
    """CPUバウンドな類似度計算"""
    return SequenceMatcher(None, text1, text2).ratio()

async def filter_results_by_diversity_async(results: List[Dict[str, Any]], threshold: float = 0.6) -> List[Dict[str, Any]]:
    """MMR風フィルタの非同期ラッパー (イベントループをブロックしない)"""
    loop = asyncio.get_running_loop()
    unique_results = []
    for doc in results:
        content = doc.get('content', '')
        is_duplicate = False
        # 既存の選択済みドキュメントと比較
        for selected_doc in unique_results:
            # 別スレッドで類似度計算を実行
            similarity = await loop.run_in_executor(
                executor,
                _compute_similarity,
                content,
                selected_doc.get('content', '')
            )
            if similarity > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_results.append(doc)
    return unique_results

def clean_json_string(json_str: str) -> str:
    """GeminiがMarkdownコードブロックを含めて返した場合のクリーニング"""
    cleaned = re.sub(r'^```json\s*', '', json_str)
    cleaned = re.sub(r'^```\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    return cleaned.strip()

# -----------------------------------------------
# AI ロジック関数群
# -----------------------------------------------

async def check_ambiguity_and_suggest_options(query: str, session_id: str) -> Dict[str, Any]:
    # 単純なルールベース判定（高速化）
    if len(query) > 10 and any(x in query for x in ["方法", "場所", "申請", "について", "教え"]):
        return {"is_ambiguous": False}

    prompt = f"""
    あなたは大学のヘルプデスクAIです。ユーザーの質問が「キーワード検索レベル」で曖昧か、
    それとも「回答可能な文章」になっているかを厳密に判定してください。
    ユーザーの質問: "{query}"
    # 指示
    - 具体的で意図が明確なら is_ambiguous: false
    - 単語のみ（例:「奨学金」「履修」）や主語・目的語不足なら is_ambiguous: true
    - candidatesには、その単語から想定される具体的な質問文を3〜4つ提案してください。
    """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = await model.generate_content_async(
            prompt,
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                response_schema=AmbiguityAnalysis
            ),
            safety_settings=SAFETY_SETTINGS
        )
        return json.loads(clean_json_string(response.text))
    except Exception as e:
        log_context(session_id, f"曖昧性判定スキップ: {e}", "warning")
        return {"is_ambiguous": False}

async def generate_search_optimized_query(user_query: str, session_id: str) -> str:
    """HyDE + Query Expansion: 専門用語補完と検索クエリ最適化"""
    prompt = f"""
    ユーザーの質問に基づいて、大学のデータベース（シラバス、学則、FAQ）から最適な情報を引き出すための「検索用キーワード」と「仮説的な回答の一部」を作成してください。
    ユーザーの質問: "{user_query}"
    # 役割
    専門用語（例: "取り消し" -> "履修中止", "休み" -> "休業期間"）への言い換えを含めてください。
    出力は検索に使用するテキストのみにしてください。
    """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = await model.generate_content_async(prompt, safety_settings=SAFETY_SETTINGS)
        optimized = response.text.strip()
        log_context(session_id, f"クエリ拡張: {user_query} -> {optimized}")
        return optimized
    except Exception:
        return user_query

async def rerank_documents_with_gemini(query: str, documents: List[Dict[str, Any]], session_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
    if not documents:
        return []

    # コンテキスト長を拡張（より正確なリランクのため）
    candidates_text = ""
    for i, doc in enumerate(documents):
        # メタデータも含めて判断材料にする
        meta = doc.get('metadata', {})
        source = meta.get('source', 'unknown')
        title = meta.get('title', 'No Title')
        # 重要な情報が後半にある場合も考慮し、2000文字まで取得
        content_snippet = doc.get('content', '')[:2000].replace('\n', ' ')
        candidates_text += f"ID:{i} [Source:{source}] [Title:{title}]\nContent: {content_snippet}\n\n"

    prompt = f"""
    あなたは優秀な検索リランカー（Re-ranker）です。
    ユーザーの質問に対して、提供されたドキュメントが「回答根拠として適切か」を0.0〜10.0点で採点してください。
    # ユーザーの質問
    {query}
    # ドキュメント候補
    {candidates_text}
    # 採点基準（品質重視）
    - 10点: 質問に対する直接的な回答が含まれており、最新かつ正確である。
    - 5-9点: 関連情報は含まれるが、部分的に推論が必要である。
    - 0-4点: キーワードは一致するが、文脈が異なる、または古い情報である。
    # 出力
    JSON形式で、各IDに対するscoreとreason（採点理由）を出力してください。
    """

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = await model.generate_content_async(
            prompt,
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                response_schema=RerankResult
            ),
            safety_settings=SAFETY_SETTINGS
        )
        result_json = json.loads(clean_json_string(response.text))
        ranked_items = result_json.get("ranked_items", [])
        reranked_docs = []
        for item in ranked_items:
            idx = int(item["id"])
            score = float(item["score"])
            if 0 <= idx < len(documents):
                doc = documents[idx]
                doc['rerank_score'] = score
                reranked_docs.append(doc)
        # スコア順ソート
        reranked_docs.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        return reranked_docs[:top_k]

    except Exception as e:
        log_context(session_id, f"リランク処理エラー: {e}", "error")
        return documents[:top_k]

# -----------------------------------------------
# メインチャットロジック (Status Update対応版)
# -----------------------------------------------

async def enhanced_chat_logic(request: Request, chat_req: ChatQuery):
    user_input = chat_req.query.strip()
    session_id = get_or_create_session_id(request)
    feedback_id = str(uuid.uuid4())
    yield f"data: {json.dumps({'feedback_id': feedback_id})}\n\n"

    try:
        # --- PHASE 1: 意図理解 ---
        yield f"data: {json.dumps({'status_message': '🤔 質問の意図を分析しています...'})}\n\n"

        # Step 1: 曖昧性チェック
        ambiguity_res = await check_ambiguity_and_suggest_options(user_input, session_id)
        if ambiguity_res.get("is_ambiguous"):
            suggestion = ambiguity_res.get("response_text", "もう少し具体的に教えていただけますか？")
            candidates = ambiguity_res.get("candidates", [])
            resp_content = suggestion
            if candidates:
                resp_content += "\n\n**もしかして:**\n" + "\n".join([f"- {c}" for c in candidates])
            yield f"data: {json.dumps({'content': resp_content})}\n\n"
            yield f"data: {json.dumps({'show_feedback': True, 'feedback_id': feedback_id})}\n\n"
            return

        # --- PHASE 2: クエリ拡張 ---
        yield f"data: {json.dumps({'status_message': '🔍 検索キーワードを最適化しています...'})}\n\n"

        # Step 2: 並列処理 (Embedding & Query Expansion)
        log_context(session_id, "検索フェーズ開始")
        # タスク定義
        task_embed_raw = asyncio.create_task(
            genai.embed_content_async(
                model=chat_req.embedding_model,
                content=user_input,
                task_type="retrieval_query"
            )
        )
        task_transform = asyncio.create_task(generate_search_optimized_query(user_input, session_id))

        # --- FAQ チェック ---
        try:
            raw_emb_res = await task_embed_raw
            raw_embedding = raw_emb_res["embedding"]
            qa_results = core_database.db_client.search_fallback_qa(
                embedding=raw_embedding,
                match_count=3
            )
            if qa_results:
                top_qa = qa_results[0]
                # 品質重視のため、FAQの一致閾値は高めに設定
                if top_qa.get('similarity', 0) >= QA_SIMILARITY_THRESHOLD:
                    task_transform.cancel()
                    resp_text = format_urls_as_links(f"よくあるご質問に回答が見つかりました。\n\n---\n{top_qa['content']}")
                    history_manager.add_to_history(session_id, "assistant", resp_text)
                    yield f"data: {json.dumps({'content': resp_text})}\n\n"
                    yield f"data: {json.dumps({'show_feedback': True, 'feedback_id': feedback_id})}\n\n"
                    return
        except Exception as e:
            log_context(session_id, f"FAQ Search Error: {e}", "warning")

        # --- Document Search ---
        search_query_text = await task_transform
        # --- PHASE 3: データベース検索 ---
        yield f"data: {json.dumps({'status_message': '📚 学内データベースを検索しています...'})}\n\n"
        try:
            opt_emb_res = await genai.embed_content_async(
                model=chat_req.embedding_model,
                content=search_query_text,
                task_type="retrieval_query"
            )
            query_embedding = opt_emb_res["embedding"]
        except Exception:
            yield f"data: {json.dumps({'content': '検索処理に失敗しました。時間をおいてお試しください。'})}\n\n"
            return

        # 検索実行
        relevant_docs = []
        try:
            # ハイブリッド検索（広めに取得）
            raw_docs = core_database.db_client.search_documents_hybrid(
                collection_name=chat_req.collection,
                query_text=search_query_text,
                query_embedding=query_embedding,
                match_count=30  # フィルタリング前は多めに確保
            )
            # 多様性フィルタ（非同期実行でブロック回避）
            unique_docs = await filter_results_by_diversity_async(raw_docs, threshold=0.7)
            # --- PHASE 4: リランク (AIによる精査) ---
            yield f"data: {json.dumps({'status_message': '🧐 文献の重要度をAIが採点中...'})}\n\n"

            # リランク実行 (Gemini)
            # 上位候補のみをLLMに渡す
            rerank_candidates = unique_docs[:12]
            if rerank_candidates:
                reranked_docs = await rerank_documents_with_gemini(
                    query=user_input,
                    documents=rerank_candidates,
                    session_id=session_id,
                    top_k=chat_req.top_k
                )
                # スコアフィルタリング
                for d in reranked_docs:
                    if d.get('rerank_score', 0) >= RERANK_SCORE_THRESHOLD:
                        relevant_docs.append(d)
        except Exception as e:
            log_context(session_id, f"Retrieval Error: {e}", "error")

        # --- 回答生成フェーズ ---
        if not relevant_docs:
            yield f"data: {json.dumps({'content': AI_NOT_FOUND_MESSAGE_USER})}\n\n"
        else:
            # --- PHASE 5: 執筆開始 ---
            yield f"data: {json.dumps({'status_message': '✍️ 回答を執筆しています...'})}\n\n"

            # コンテキスト構築（引用番号の付与）
            context_text = ""
            sources_map = {} # {1: "履修ガイド p.10", 2: ...}
            for idx, d in enumerate(relevant_docs, 1):
                content = d.get('content', '')
                source = d.get('metadata', {}).get('source', '不明')
                # マッピング保存
                sources_map[idx] = source
                # プロンプト用コンテキスト
                if len(context_text) + len(content) < MAX_CONTEXT_CHAR_LENGTH:
                    context_text += f"<document index='{idx}' source='{source}'>\n{content}\n</document>\n\n"

            # プロンプトの質を高める（Chain of Thought誘導）
            system_prompt = f"""
            あなたは札幌学院大学の公式学生サポートAIです。
            提供された<context>内の情報**のみ**を使用して、ユーザーの質問に正確かつ論理的に回答してください。

            # 回答作成ルール
            1. **根拠の明示**: 回答内の事実には、必ず情報の出所となるドキュメント番号を `[1]` `[2]` の形式で文中に付記してください。
            - 良い例: 「履修登録の修正期間は4月15日までです[1]。」
            - 悪い例: 出所を書かない、または文末にまとめるだけ。
            2. **正確性優先**: コンテキストに答えがない場合は、正直に「情報が見つかりません」と答えてください。推測で答えることは禁止です。
            3. **トーン**: 丁寧で親しみやすい、しかし事務的に正確な「です・ます」調。
            4. **構造**: 読みやすいように箇条書きや太字を活用してください。
            <context>
            {context_text}
            </context>
            """
            # 参照元リストの作成（回答後に付与するため）
            references_text = "\n\n---\n**参考資料:**\n" + "\n".join([f"- [{k}] {v}" for k, v in sources_map.items()])

            user_prompt = f"質問: {user_input}"
            try:
                model = genai.GenerativeModel(chat_req.model)
                stream = await model.generate_content_async(
                    [system_prompt, user_prompt],
                    stream=True,
                    safety_settings=SAFETY_SETTINGS
                )
                full_response = ""
                async for chunk in stream:
                    if chunk.text:
                        text_chunk = chunk.text
                        full_response += text_chunk
                        # ストリーミング中はそのまま流す
                        yield f"data: {json.dumps({'content': text_chunk})}\n\n"
                # 回答完了後、情報が見つからなかった場合以外は参照元を追記
                if AI_NOT_FOUND_TOKEN not in full_response and "情報が見つかりません" not in full_response:
                     yield f"data: {json.dumps({'content': references_text})}\n\n"
                     # 履歴には参照元付きで保存
                     history_manager.add_to_history(session_id, "assistant", full_response + references_text)
                else:
                    history_manager.add_to_history(session_id, "assistant", full_response)

            except Exception as e:
                log_context(session_id, f"Generation Error: {e}", "error")
                yield f"data: {json.dumps({'content': '回答生成中にエラーが発生しました。'})}\n\n"

    except Exception as e:
        log_context(session_id, f"Critical Error: {e}", "error")
        yield f"data: {json.dumps({'content': 'システムエラーが発生しました。管理者に連絡してください。'})}\n\n"
    finally:
        yield f"data: {json.dumps({'show_feedback': True, 'feedback_id': feedback_id})}\n\n"