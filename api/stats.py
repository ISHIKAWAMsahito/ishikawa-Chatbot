"""
api/stats.py

管理者向け統計・AI相談API。
prefix: /api/admin/stats  (main.py で登録済み)

【エンドポイント一覧】
GET  /data                  - フィードバック履歴の取得
GET  /vectorize/status      - ベクトル化の進捗確認  ← 必ず /vectorize より先に定義
POST /vectorize             - 一括ベクトル化
POST /chat_analysis         - ベクトル検索 × LLM 相談 (SSE)
"""
import logging
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
# 2. ベクトル化の進捗確認  ← /vectorize より必ず先に定義する
# ==================================================================

@router.get("/vectorize/status", summary="ベクトル化の進捗を確認")
async def vectorize_status(user: dict = Depends(require_auth)):
    """
    chat_logs と anonymous_comments のベクトル化進捗を返す。
    テーブルが存在しない場合でも 200 を返す（total=0 扱い）。
    """
    def _count(table: str, null_filter: bool = False):
        try:
            q = db_client.client.table(table).select("id", count="exact", head=True)
            if null_filter:
                q = q.is_("embedding", "null")
            return q.execute().count or 0
        except Exception:
            return 0

    logs_total    = _count("chat_logs")
    logs_pending  = _count("chat_logs",          null_filter=True)
    cm_total      = _count("anonymous_comments")
    cm_pending    = _count("anonymous_comments", null_filter=True)

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
    """embedding が未設定のレコードを最大 limit 件ベクトル化する"""
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
    職員・教員からの質問に対し、
    chat_logs と anonymous_comments をベクトル検索して
    関連性の高いログのみをコンテキストに LLM が分析する。
    ベクトルがまだない場合はフォールバックで直近ログを使用する。
    """
    logger.info(f"Admin analysis query: {request.query}")

    try:
        # ① クエリをベクトル化
        query_embedding = await llm_service.get_embedding(request.query)

        # ② chat_logs をベクトル検索（失敗時は直近20件にフォールバック）
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

        # ③ anonymous_comments をベクトル検索
        comment_context_parts: list[str] = []
        try:
            cm_res = db_client.client.rpc(
                "match_anonymous_comments",
                {
                    "p_query_embedding": query_embedding,
                    "p_match_threshold": 0.35,
                    "p_match_count":     10,
                },
            ).execute()
            for i, row in enumerate(cm_res.data or [], 1):
                ts      = (row.get("created_at") or "")[:10]
                rating  = row.get("rating", "-")
                comment = row.get("comment", "")
                sim     = row.get("similarity", 0.0)
                comment_context_parts.append(
                    f"[意見{i} | {ts} | 評価:{rating} | 類似度:{sim:.2f}]\n{comment}"
                )
        except Exception as ce:
            logger.warning(f"anonymous_comments vector search failed: {ce}")

        # ④ データがまったくない場合
        if not log_context_parts and not comment_context_parts:
            async def _empty():
                yield (
                    "分析対象となるデータがまだありません。\n\n"
                    "まずは学生がチャット画面で質問を行い、"
                    "「ベクトル化」ボタンでデータを登録してください。"
                )
            return StreamingResponse(_empty(), media_type="text/event-stream")

        # ⑤ プロンプト構築
        log_block     = "\n\n".join(log_context_parts)     if log_context_parts     else "（該当するチャットログなし）"
        comment_block = "\n\n".join(comment_context_parts) if comment_context_parts else "（該当する意見・要望なし）"

        user_prompt = (
            f"【職員・教員からの相談】\n{request.query}\n\n"
            f"【関連チャットログ（学生 ↔ AI）】\n{log_block}\n\n"
            f"【関連する学生からの意見・要望】\n{comment_block}"
        )

        # ⑥ SSEストリーミング
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