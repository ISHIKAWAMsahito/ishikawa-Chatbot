"""
services/vectorize_logs.py

chat_logs および anonymous_comments のベクトル化・一括処理サービス。

【注意】
chat_logs.embedding カラムの次元数と、Gemini 埋め込みモデルの出力次元数が
一致しない場合（例: DBが vector(1536)、モデルが 768 or 3072 を出力）、
PATCH が 400 エラーになります。
→ エラー時は WARNING に留めて処理継続。
→ 管理者が stats.html の「一括ベクトル化」ボタンで手動実行することを推奨。
"""
import asyncio
import logging
from typing import Optional

import google.generativeai as genai

from core.config import GEMINI_API_KEY
from core.database import db_client

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "models/gemini-embedding-001"

genai.configure(api_key=GEMINI_API_KEY)


async def _embed_text(text: str) -> Optional[list[float]]:
    """テキストをGemini APIでベクトル化する（非同期）。"""
    if not text or not text.strip():
        return None
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text.strip()[:8000],
                task_type="retrieval_document",
            ),
        )
        return result["embedding"]
    except Exception as e:
        logger.warning(f"Embedding generation failed: {e}")
        return None


async def vectorize_chat_log(
    log_id: str, user_query: str, ai_response: str
) -> bool:
    """
    chat_logs の1件に対してembeddingを付与する。

    Returns:
        True: 成功 / False: 失敗（ログ出力済み）
    """
    combined = f"質問: {user_query}\n回答の要点: {ai_response[:500]}"
    embedding = await _embed_text(combined)
    if embedding is None:
        return False

    try:
        db_client.client.table("chat_logs") \
            .update({"embedding": embedding}) \
            .eq("id", log_id) \
            .execute()
        logger.debug(f"chat_log {log_id} vectorized.")
        return True
    except Exception as e:
        err_msg = str(e)
        # ★ 次元数不一致は頻発する既知エラーなので WARNING に留める
        if "dimensions" in err_msg or "22000" in err_msg:
            logger.warning(
                f"chat_log embedding skipped (dimension mismatch): "
                f"ID={log_id}. DBスキーマの vector 次元数を確認してください。"
            )
        else:
            logger.error(f"Failed to update chat_log embedding ({log_id}): {e}")
        return False


async def bulk_vectorize_chat_logs(limit: int = 200) -> dict:
    """embeddingが未設定の chat_logs を最大 limit 件ベクトル化する。"""
    try:
        res = (
            db_client.client.table("chat_logs")
            .select("id, user_query, ai_response")
            .is_("embedding", "null")
            .limit(limit)
            .execute()
        )
        rows = res.data or []
    except Exception as e:
        logger.error(f"bulk_vectorize_chat_logs fetch error: {e}")
        return {"processed": 0, "succeeded": 0, "failed": 0}

    succeeded = 0
    failed = 0
    for row in rows:
        ok = await vectorize_chat_log(
            log_id=row["id"],
            user_query=row.get("user_query", ""),
            ai_response=row.get("ai_response", ""),
        )
        if ok:
            succeeded += 1
        else:
            failed += 1
        await asyncio.sleep(0.3)

    logger.info(f"bulk_vectorize_chat_logs: {succeeded} ok / {failed} failed")
    return {"processed": len(rows), "succeeded": succeeded, "failed": failed}


async def vectorize_comment(comment_id: str, comment_text: str) -> bool:
    """anonymous_comments の1件にembeddingを付与する。"""
    embedding = await _embed_text(comment_text)
    if embedding is None:
        return False

    try:
        db_client.client.table("anonymous_comments") \
            .update({"embedding": embedding}) \
            .eq("id", comment_id) \
            .execute()
        logger.debug(f"comment {comment_id} vectorized.")
        return True
    except Exception as e:
        err_msg = str(e)
        if "dimensions" in err_msg or "22000" in err_msg:
            logger.warning(
                f"comment embedding skipped (dimension mismatch): ID={comment_id}."
            )
        else:
            logger.error(f"Failed to update comment embedding ({comment_id}): {e}")
        return False


async def bulk_vectorize_comments(limit: int = 200) -> dict:
    """embeddingが未設定の anonymous_comments を最大 limit 件ベクトル化する。"""
    try:
        res = (
            db_client.client.table("anonymous_comments")
            .select("id, comment")
            .is_("embedding", "null")
            .not_.is_("comment", "null")
            .limit(limit)
            .execute()
        )
        rows = res.data or []
    except Exception as e:
        logger.error(f"bulk_vectorize_comments fetch error: {e}")
        return {"processed": 0, "succeeded": 0, "failed": 0}

    succeeded = 0
    failed = 0
    for row in rows:
        comment_text = row.get("comment", "").strip()
        if not comment_text:
            continue
        ok = await vectorize_comment(row["id"], comment_text)
        if ok:
            succeeded += 1
        else:
            failed += 1
        await asyncio.sleep(0.3)

    logger.info(f"bulk_vectorize_comments: {succeeded} ok / {failed} failed")
    return {"processed": len(rows), "succeeded": succeeded, "failed": failed}