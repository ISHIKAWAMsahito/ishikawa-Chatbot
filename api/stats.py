"""
api/stats.py

管理者向け統計・AI相談API。
prefix: /api/admin/stats  (main.py で登録済み)

【エンドポイント一覧】
GET  /data                  - フィードバック履歴の取得
GET  /vectorize/status      - ベクトル化の進捗確認  ← /vectorize より先に定義
POST /vectorize             - 一括ベクトル化
POST /chat_analysis         - ベクトル検索 × LLM 相談 (SSE)
"""
import logging
import re
from typing import AsyncGenerator, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.database import db_client
from core.dependencies import require_auth
from models.schemas import AnalysisQuery, FeedbackItem
from services.llm import LLMService
from services import prompts

logger = logging.getLogger(__name__)
router = APIRouter()
llm_service = LLMService()


# ------------------------------------------------------------------
# Pydantic
# ------------------------------------------------------------------

class VectorizeRequest(BaseModel):
    target: str = Field("both", description="'chat_logs' | 'comments' | 'both'")
    limit: int  = Field(200, ge=1, le=1000)


# ==================================================================
# 共通: comment カラムから表示用テキストを抽出
# ==================================================================

def _summarize_comment(comment: str, rating: str) -> str:
    """
    anonymous_comments.comment（Q: / A: 形式）から
    分析コンテキスト用の短いテキストを生成する。
    """
    text = (comment or "").strip()
    rating_label = {"good": "✅", "bad": "❌"}.get(rating, "−")

    q_match = re.search(r"Q:\s*(.*?)(?=\nA:|\Z)", text, re.DOTALL)
    a_match = re.search(r"A:\s*(.*)", text, re.DOTALL)

    q = (q_match.group(1).strip() if q_match else "").strip()
    a = (a_match.group(1).strip() if a_match else "").strip()

    if q:
        a_short = (a[:150].split("\n")[0]) if a else "（回答なし）"
        return f"{rating_label} Q: {q} → A要約: {a_short}"
    elif a:
        return f"{rating_label} {a[:200]}"
    else:
        return f"{rating_label} {text[:200]}"


# ==================================================================
# 1. 統計データ取得
# ==================================================================

@router.get("/data", response_model=List[FeedbackItem])
async def get_stats_data(user: dict = Depends(require_auth)):
    """anonymous_comments の最新100件を返す"""
    try:
        response = (
            db_client.client
            .table("anonymous_comments")
            .select("*")
            .order("created_at", desc=True)
            .limit(100)
            .execute()
        )
        return response.data
    except Exception as e:
        logger.error(f"Error fetching stats data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="データの取得に失敗しました")


# ==================================================================
# 2. ベクトル化の進捗確認  ← /vectorize より先に定義
# ==================================================================

@router.get("/vectorize/status", summary="ベクトル化の進捗を確認")
async def vectorize_status(user: dict = Depends(require_auth)):
    """
    chat_logs と anonymous_comments のベクトル化進捗を返す。
    """
    def _count(table: str, null_filter: bool = False) -> int:
        try:
            q = db_client.client.table(table).select("id", count="exact", head=True)
            if null_filter:
                q = q.is_("embedding", "null")
            return q.execute().count or 0
        except Exception:
            return 0

    logs_total   = _count("chat_logs")
    logs_pending = _count("chat_logs",          null_filter=True)
    cm_total     = _count("anonymous_comments")
    cm_pending   = _count("anonymous_comments", null_filter=True)

    return {
        "chat_logs": {
            "total":      logs_total,
            "pending":    logs_pending,
            "vectorized": logs_total - logs_pending,
        },
        "anonymous_comments": {
            "total":      cm_total,
            "pending":    cm_pending,
            "vectorized": cm_total - cm_pending,
        },
    }


# ==================================================================
# 3. 一括ベクトル化
# ==================================================================

@router.post("/vectorize", summary="chat_logs / anonymous_comments を一括ベクトル化")
async def bulk_vectorize(
    request: VectorizeRequest,
    user: dict = Depends(require_auth),
):
    from services.vectorize_logs import bulk_vectorize_chat_logs, bulk_vectorize_comments

    if request.target not in ("chat_logs", "comments", "both"):
        raise HTTPException(
            status_code=400,
            detail="target は 'chat_logs' | 'comments' | 'both' を指定してください",
        )

    results: dict = {}
    if request.target in ("chat_logs", "both"):
        results["chat_logs"] = await bulk_vectorize_chat_logs(limit=request.limit)
    if request.target in ("comments", "both"):
        results["anonymous_comments"] = await bulk_vectorize_comments(limit=request.limit)

    return {"status": "ok", "results": results}


# ==================================================================
# 4. AI相談: ベクトル検索 × LLM (SSEストリーミング)
# ==================================================================

@router.post("/chat_analysis", summary="対話ログに基づく相談・分析")
async def chat_with_logs(
    request: AnalysisQuery,
    user: dict = Depends(require_auth),
):
    """
    職員・教員の質問をベクトル検索して関連ログを収集し、
    LLM（Gemini 2.5 Flash）がSSEでストリーミング回答する。

    データソース:
      ① chat_logs            … 学生↔AI の会話ログ
      ② anonymous_comments   … Q&A形式 + good/bad 評価付きフィードバック
    """
    logger.info(f"Admin analysis query: {request.query}")

    try:
        query_embedding = await llm_service.get_embedding(request.query)

        # ── ① chat_logs ベクトル検索 ──────────────────────────────
        log_context_parts: list[str] = []
        try:
            log_res = db_client.client.rpc(
                "match_chat_logs",
                {
                    "p_query_embedding": query_embedding,
                    "p_match_threshold": 0.35,
                    "p_match_count":     20,
                },
            ).execute()
            for i, row in enumerate(log_res.data or [], 1):
                ts  = (row.get("created_at") or "")[:10]
                q   = row.get("user_query", "")
                a   = (row.get("ai_response") or "")[:200]
                sim = row.get("similarity", 0.0)
                log_context_parts.append(
                    f"[ログ{i} | {ts} | 類似度:{sim:.2f}]\nQ: {q}\nA要約: {a}"
                )
        except Exception as ve:
            logger.warning(f"chat_logs vector search failed, fallback: {ve}")
            try:
                fb = (
                    db_client.client.table("chat_logs")
                    .select("created_at, user_query, ai_response")
                    .order("created_at", desc=True)
                    .limit(20)
                    .execute()
                )
                for i, row in enumerate(fb.data or [], 1):
                    ts = (row.get("created_at") or "")[:10]
                    q  = row.get("user_query", "")
                    a  = (row.get("ai_response") or "")[:200]
                    log_context_parts.append(f"[ログ{i} | {ts}]\nQ: {q}\nA要約: {a}")
            except Exception as fe:
                logger.warning(f"chat_logs fallback also failed: {fe}")

        # ── ② anonymous_comments ベクトル検索（rating付き）─────────
        comment_context_parts: list[str] = []
        bad_questions: list[str] = []   # bad評価の質問を別途収集
        try:
            cm_res = db_client.client.rpc(
                "match_anonymous_comments",
                {
                    "p_query_embedding": query_embedding,
                    "p_match_threshold": 0.35,
                    "p_match_count":     15,
                },
            ).execute()
            for i, row in enumerate(cm_res.data or [], 1):
                ts      = (row.get("created_at") or "")[:10]
                rating  = row.get("rating") or ""
                comment = row.get("comment", "") or ""
                sim     = row.get("similarity", 0.0)
                summary = _summarize_comment(comment, rating)
                comment_context_parts.append(
                    f"[FB{i} | {ts} | 類似度:{sim:.2f}]\n{summary}"
                )
                # bad評価の質問文を抽出
                if rating == "bad":
                    q_match = re.search(r"Q:\s*(.*?)(?=\nA:|\Z)", comment, re.DOTALL)
                    q_text = (q_match.group(1).strip() if q_match else "").strip()
                    if q_text:
                        bad_questions.append(q_text)
        except Exception as ce:
            logger.warning(f"anonymous_comments vector search failed: {ce}")

        # ── ③ データなし ────────────────────────────────────────────
        if not log_context_parts and not comment_context_parts:
            async def _empty():
                yield (
                    "分析対象となるデータがまだありません。\n\n"
                    "まずは「⚡ 未処理を一括ベクトル化」ボタンでデータを登録してください。"
                )
            return StreamingResponse(_empty(), media_type="text/event-stream")

        # ── ④ プロンプト構築 ──────────────────────────────────────
        log_block     = "\n\n".join(log_context_parts)     or "（該当するチャットログなし）"
        comment_block = "\n\n".join(comment_context_parts) or "（該当するフィードバックなし）"

        bad_block = ""
        if bad_questions:
            bad_block = (
                "\n\n【❌ bad評価の質問一覧（AIが答えられなかった）】\n"
                + "\n".join(f"  - {q}" for q in bad_questions)
            )

        user_prompt = (
            f"【職員・教員からの相談】\n{request.query}\n\n"
            f"【関連チャットログ（学生 ↔ AI）】\n{log_block}\n\n"
            f"【関連するフィードバック（✅good / ❌bad 評価付き）】\n"
            f"{comment_block}{bad_block}"
        )

        # ── ⑤ SSEストリーミング ──────────────────────────────────
        return StreamingResponse(
            _generate_stream(user_prompt),
            media_type="text/event-stream",
        )

    except Exception as e:
        logger.error(f"Analysis Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"分析処理中にエラーが発生しました: {e}")


# ------------------------------------------------------------------
# 内部: LLM ストリーム生成
# ------------------------------------------------------------------

async def _generate_stream(user_prompt: str) -> AsyncGenerator[str, None]:
    import google.generativeai as genai
    from services.llm import ROBUST_SAFETY_SETTINGS

    try:
        model = genai.GenerativeModel(
            "models/gemini-2.5-flash",
            system_instruction=prompts.LOG_ANALYSIS_ADVISOR,
        )
        response = await model.generate_content_async(
            user_prompt,
            stream=True,
            safety_settings=ROBUST_SAFETY_SETTINGS,
        )
        async for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        logger.error(f"Stream generation error: {e}")
        yield f"\n\nエラーが発生しました: {e}"