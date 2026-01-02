import logging
import uuid
import json
import asyncio
import re
import random
from collections import defaultdict
from typing import List, Dict, Any, AsyncGenerator, Optional
import typing_extensions as typing  # 型定義用

from fastapi import Request, HTTPException
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
# 定数・設定値 (Configuration)
# -----------------------------------------------
# 検索・マッチング閾値
STRICT_THRESHOLD = 0.80
QA_SIMILARITY_THRESHOLD = 0.95  # FAQ即答ライン
RERANK_SCORE_THRESHOLD = 6.0    # 10点満点中の採用ライン

# 日本語1文字≒1〜1.5トークン換算でも余裕を持たせる
MAX_CONTEXT_CHAR_LENGTH = 100000

# 履歴保持数
MAX_HISTORY_LENGTH = 20

# AI応答制御トークン
AI_NOT_FOUND_TOKEN = "[[NO_RELEVANT_INFO_FOUND]]"
AI_NOT_FOUND_MESSAGE_USER = (
    "ご質問いただいた内容については、関連する情報が見つかりませんでした。"
    "お手数ですが、大学の公式サイトをご確認いただくか、窓口までお問い合わせください。"
)

# セーフティ設定（誤検知によるブロックを防ぎつつ安全性を確保）
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}

# -----------------------------------------------
# 構造化出力用の型定義 (Schemas)
# -----------------------------------------------
class AmbiguityAnalysis(typing.TypedDict):
    is_ambiguous: bool
    response_text: str
    candidates: List[str]

class RerankItem(typing.TypedDict):
    id: int
    score: float

class RerankResult(typing.TypedDict):
    ranked_items: List[RerankItem]

# -----------------------------------------------
# ユーティリティ関数
# -----------------------------------------------
def log_context(session_id: str, message: str, level: str = "info"):
    """セッションID付きでログを出力するラッパー関数"""
    msg = f"[Session: {session_id}] {message}"
    if level == "error":
        logging.error(msg)
    elif level == "warning":
        logging.warning(msg)
    else:
        logging.info(msg)

class ChatHistoryManager:
    """チャット履歴管理 (インメモリ)"""
    def __init__(self):
        self._histories: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    def add_to_history(self, session_id: str, role: str, content: str):
        # メモリ節約のため一時的に無効化する場合はここでreturn
        # return 
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

def filter_results_by_diversity(results: List[Dict[str, Any]], threshold: float = 0.6) -> List[Dict[str, Any]]:
    """MMR風フィルタ: 内容が酷似している重複ドキュメントを排除"""
    unique_results = []
    for doc in results:
        content = doc.get('content', '')
        is_duplicate = False
        for selected_doc in unique_results:
            similarity = SequenceMatcher(None, content, selected_doc.get('content', '')).ratio()
            if similarity > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_results.append(doc)
    return unique_results

# -----------------------------------------------
# AI ロジック関数群 (Structured Outputs 対応)
# -----------------------------------------------

async def check_ambiguity_and_suggest_options(query: str, session_id: str) -> Dict[str, Any]:
    """質問の曖昧性を判定し、必要なら候補を提示する"""
    # 1. ルールベースによる高速判定 (コスト削減)
    if len(query) > 15 or any(w in query for w in ["方法", "場所", "申請", "いつ", "どこ", "何", "？", "?"]):
        return {"is_ambiguous": False}

    prompt = f"""
    あなたは大学のヘルプデスクAIです。ユーザーの質問が「単語のみ」などで曖昧か判定してください。
    ユーザーの質問: "{query}"
    # 指示
    - 質問が具体的（文脈がある、複合語である）なら is_ambiguous: false
    - 質問が漠然としているなら is_ambiguous: true とし、candidates に予測される質問意図を3つ、最後に「その他」を含めて計4つ挙げてください。
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
        return json.loads(response.text)
    except Exception as e:
        log_context(session_id, f"曖昧性判定エラー: {e}", "warning")
        return {"is_ambiguous": False}

async def generate_search_optimized_query(user_query: str, session_id: str) -> str:
    """HyDE / クエリ拡張: 検索ヒット率を高めるためのクエリ変換"""
    if len(user_query) < 5:
        return user_query

    prompt = f"""
    ユーザーの質問に対して、大学のデータベース検索で最適なドキュメントがヒットするような「検索用クエリ」を生成してください。
    質問の意図を汲み取り、関連するキーワードや具体的な表現を補完してください。
    ユーザーの質問: "{user_query}"
    出力: 検索用クエリのみを出力（余計な説明は不要）
    """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = await model.generate_content_async(prompt, safety_settings=SAFETY_SETTINGS)
        optimized = response.text.strip()
        log_context(session_id, f"クエリ変換: {user_query} -> {optimized}")
        return optimized
    except Exception as e:
        log_context(session_id, f"クエリ変換エラー: {e}", "warning")
        return user_query

async def rerank_documents_with_gemini(query: str, documents: List[Dict[str, Any]], session_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Geminiによるリランク (Structured Outputs使用)"""
    if not documents:
        return []
    if len(documents) == 1:
        return documents

    log_context(session_id, f"リランク開始: {len(documents)}件の候補を評価")

    # プロンプト用テキスト生成
    candidates_text = ""
    for i, doc in enumerate(documents):
        # 判定に必要な冒頭部分とメタデータのみを渡す（トークン節約と速度向上）
        content_snippet = doc.get('content', '')[:800].replace('\n', ' ')
        source = doc.get('metadata', {}).get('source', 'unknown')
        candidates_text += f"ID:{i} Source:{source} Content:{content_snippet}\n\n"

    prompt = f"""
    あなたは検索エンジンのRe-rankerです。ユーザーの質問に対して、各ドキュメントの関連度を0〜10点で採点してください。
    # ユーザーの質問
    {query}
    # ドキュメント候補
    {candidates_text}
    # 採点基準
    - 質問の意図（学部、手続き、期限など）に合致しているか。
    - まったく関係ないドキュメントは 0点 にすること。
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
        result_json = json.loads(response.text)
        ranked_items = result_json.get("ranked_items", [])
        # スコアのマッピングと並べ替え
        reranked_docs = []
        for item in ranked_items:
            idx = int(item["id"])
            score = float(item["score"])
            if 0 <= idx < len(documents):
                doc = documents[idx]
                doc['rerank_score'] = score
                reranked_docs.append(doc)
        # スコア降順にソート
        reranked_docs.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        return reranked_docs[:top_k]

    except Exception as e:
        log_context(session_id, f"リランク処理エラー: {e}", "error")
        # エラー時は元の順序で返す
        return documents[:top_k]

# -----------------------------------------------
# メインチャットロジック
# -----------------------------------------------

async def enhanced_chat_logic(request: Request, chat_req: ChatQuery):
    """
    FAQ検索 -> ドキュメント検索 (Hybrid) -> Rerank -> 回答生成
    のパイプラインを実行する
    """
    user_input = chat_req.query.strip()
    session_id = get_or_create_session_id(request)
    feedback_id = str(uuid.uuid4())
    # クライアントへ初期レスポンス（フィードバックID）
    yield f"data: {json.dumps({'feedback_id': feedback_id})}\n\n"

    # ヘルスチェック
    if not all([core_database.db_client, GEMINI_API_KEY]):
        log_context(session_id, "DBクライアントまたはAPIキー未設定", "error")
        yield f"data: {json.dumps({'content': 'システムエラー: 管理者にお問い合わせください。'})}\n\n"
        return

    try:
        # Step 0: 曖昧性チェック
        ambiguity_res = await check_ambiguity_and_suggest_options(user_input, session_id)
        if ambiguity_res.get("is_ambiguous"):
            suggestion = ambiguity_res.get("response_text", "もう少し具体的に教えていただけますか？")
            candidates = ambiguity_res.get("candidates", [])
            if candidates:
                suggestion += "\n\n" + "\n".join([f"・{c}" for c in candidates])
            yield f"data: {json.dumps({'content': suggestion})}\n\n"
            yield f"data: {json.dumps({'show_feedback': True, 'feedback_id': feedback_id})}\n\n"
            return

        # Step 1: 並列処理開始 (クエリ変換 & FAQ埋め込み)
        log_context(session_id, "検索プロセス開始")
        # タスクA: 生のクエリでFAQ検索用ベクトル作成
        task_embed_raw = asyncio.create_task(
            genai.embed_content_async(
                model=chat_req.embedding_model,
                content=user_input,
                task_type="retrieval_query"
            )
        )
        # タスクB: ドキュメント検索用にクエリを最適化
        task_transform = asyncio.create_task(generate_search_optimized_query(user_input, session_id))

        # --- FAQ (QA Database) チェック ---
        try:
            raw_emb_res = await task_embed_raw
            raw_embedding = raw_emb_res["embedding"]
            qa_results = core_database.db_client.search_fallback_qa(
                embedding=raw_embedding,
                match_count=3
            )
            # FAQ即決ロジック
            if qa_results:
                top_qa = qa_results[0]
                sim = top_qa.get('similarity', 0)
                if sim >= QA_SIMILARITY_THRESHOLD:
                    log_context(session_id, f"FAQ完全一致 (Sim: {sim:.4f})")
                    task_transform.cancel() # クエリ変換キャンセル
                    resp_text = format_urls_as_links(f"よくあるご質問に見つかりました。\n\n---\n{top_qa['content']}")
                    history_manager.add_to_history(session_id, "user", user_input)
                    history_manager.add_to_history(session_id, "assistant", resp_text)
                    yield f"data: {json.dumps({'content': resp_text})}\n\n"
                    yield f"data: {json.dumps({'show_feedback': True, 'feedback_id': feedback_id})}\n\n"
                    return
        except Exception as e:
            log_context(session_id, f"FAQチェック中の警告: {e}", "warning")

        # --- Document RAG (Hybrid Search) ---
        search_query_text = await task_transform
        # 拡張クエリのベクトル化
        try:
            opt_emb_res = await genai.embed_content_async(
                model=chat_req.embedding_model,
                content=search_query_text,
                task_type="retrieval_query"
            )
            query_embedding = opt_emb_res["embedding"]
        except Exception as e:
            log_context(session_id, f"Embeddingエラー: {e}", "error")
            yield f"data: {json.dumps({'content': '検索処理中にエラーが発生しました。'})}\n\n"
            return

        # 検索実行
        relevant_docs = []
        try:
            # 1. ハイブリッド検索 (or Vector Fallback)
            raw_docs = core_database.db_client.search_documents_hybrid(
                collection_name=chat_req.collection,
                query_text=search_query_text,
                query_embedding=query_embedding,
                match_count=20 # 多めに取得
            )
            # 2. 多様性フィルタ (MMR)
            unique_docs = filter_results_by_diversity(raw_docs, threshold=0.7)
            # 3. リランク (上位10件程度に絞ってからGeminiに投げる)
            rerank_candidates = unique_docs[:10]
            if len(rerank_candidates) > 0:
                reranked_docs = await rerank_documents_with_gemini(
                    query=user_input, # 生のクエリで判定させる
                    documents=rerank_candidates,
                    session_id=session_id,
                    top_k=chat_req.top_k
                )
                # 閾値による足切り
                for d in reranked_docs:
                    score = d.get('rerank_score', 0)
                    if score >= RERANK_SCORE_THRESHOLD:
                        relevant_docs.append(d)
                    else:
                        log_context(session_id, f"ドキュメント却下 Score:{score}")

        except Exception as e:
            log_context(session_id, f"ドキュメント検索エラー: {e}", "error")

        # --- 回答生成フェーズ ---
        if not relevant_docs:
            log_context(session_id, "有効なドキュメントなし")
            yield f"data: {json.dumps({'content': AI_NOT_FOUND_MESSAGE_USER})}\n\n"
        else:
            # コンテキスト構築
            context_text = ""
            used_sources = []
            for d in relevant_docs:
                content = d.get('content', '')
                source = d.get('metadata', {}).get('source', '不明')
                # コンテキストサイズ上限チェック
                if len(context_text) + len(content) < MAX_CONTEXT_CHAR_LENGTH:
                    context_text += f"<document source='{source}'>\n{content}\n</document>\n\n"
                    used_sources.append(source)
            system_prompt = f"""
            あなたは札幌学院大学の学生サポートAIです。
            以下の<context>情報を基に、ユーザーの質問に回答してください。
            # 重要ルール
            1. 事実に基づかない回答は禁止です。情報がない場合は「{AI_NOT_FOUND_TOKEN}」と出力してください。
            2. 文体は親しみやすい「です・ます」調にしてください。
            3. 参照した情報源（source）がある場合、回答中に自然に言及するか、末尾にまとめてください。
            <context>
            {context_text}
            </context>
            """
            user_prompt = f"質問: {user_input}"
            log_context(session_id, f"回答生成開始: ソース={list(set(used_sources))}")
            model = genai.GenerativeModel(chat_req.model)
            full_response = ""
            try:
                # ストリーミング生成
                stream = await model.generate_content_async(
                    [system_prompt, user_prompt],
                    stream=True,
                    safety_settings=SAFETY_SETTINGS
                )
                async for chunk in stream:
                    if chunk.text:
                        full_response += chunk.text
                # 回答なしトークンのチェック
                if AI_NOT_FOUND_TOKEN in full_response:
                    yield f"data: {json.dumps({'content': AI_NOT_FOUND_MESSAGE_USER})}\n\n"
                else:
                    formatted_resp = format_urls_as_links(full_response)
                    history_manager.add_to_history(session_id, "user", user_input)
                    history_manager.add_to_history(session_id, "assistant", formatted_resp)
                    yield f"data: {json.dumps({'content': formatted_resp})}\n\n"

            except Exception as e:
                log_context(session_id, f"生成エラー: {e}", "error")
                yield f"data: {json.dumps({'content': '申し訳ありません。回答の生成中にエラーが発生しました。'})}\n\n"

    except Exception as e:
        log_context(session_id, f"予期せぬクリティカルエラー: {e}", "error")
        yield f"data: {json.dumps({'content': 'システムエラーが発生しました。'})}\n\n"
    finally:
        # フィードバックUIの表示トリガー
        yield f"data: {json.dumps({'show_feedback': True, 'feedback_id': feedback_id})}\n\n"

# -----------------------------------------------
# 分析用ロジック (管理者機能)
# -----------------------------------------------
async def analyze_feedback_trends(logs: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
    if not logs:
        yield f"data: {json.dumps({'content': '分析対象データがありません。'})}\n\n"
        return

    # ログデータの整形（AIのコンテキスト消費を抑えるため要約）
    formatted_logs = ""
    for log in logs[:50]: # 直近50件に制限
        rating = log.get('rating', '-')
        comment = log.get('comment', '')[:200].replace('\n', ' ')
        formatted_logs += f"- 評価:{rating} | 内容:{comment}\n"

    prompt = f"""
    あなたはシステム運用コンサルタントです。以下のチャットボット利用ログを分析し、Markdown形式でレポートを作成してください。
    # 分析対象データ
    {formatted_logs}
    # 出力項目
    1. ユーザーの主な関心事（トレンド）
    2. 低評価の原因と改善策
    3. 次のアクションプラン
    """

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        stream = await model.generate_content_async(prompt, stream=True)
        async for chunk in stream:
            if chunk.text:
                yield f"data: {json.dumps({'content': chunk.text})}\n\n"
    except Exception as e:
        logging.error(f"分析機能エラー: {e}")
        yield f"data: {json.dumps({'content': '分析中にエラーが発生しました。'})}\n\n"