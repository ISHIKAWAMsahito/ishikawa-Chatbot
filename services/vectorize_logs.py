"""
services/vectorize_logs.py

chat_logs および anonymous_comments の
ベクトル化・一括処理サービス。

【役割】
- 新規ログ保存時にバックグラウンドでembeddingを付与する
- 管理者APIから一括ベクトル化を実行する
"""
import asyncio
import logging
from typing import Optional

import google.generativeai as genai

from core.config import GEMINI_API_KEY
from core.database import db_client

logger = logging.getLogger(__name__)

# Gemini埋め込みモデル（config.pyと統一）
EMBEDDING_MODEL = "models/gemini-embedding-001"

genai.configure(api_key=GEMINI_API_KEY)


# ------------------------------------------------------------------
# 内部ヘルパー
# ------------------------------------------------------------------

async def _embed_text(text: str) -> Optional[list[float]]:
    """
    テキストをGemini APIでベクトル化する（非同期）。
    失敗時はNoneを返す（呼び出し元でスキップ扱い）。
    """
    if not text or not text.strip():
        return None
    try:
        # Gemini SDK の embed_content は同期なのでスレッドで実行
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text.strip()[:8000],  # 上限対策
                task_type="retrieval_document"
            )
        )
        return result["embedding"]
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return None


# ------------------------------------------------------------------
# chat_logs: 1件ベクトル化（ログ保存直後に呼ぶ）
# ------------------------------------------------------------------

async def vectorize_chat_log(log_id: str, user_query: str, ai_response: str) -> bool:
    """
    chat_logs の1件に対してembeddingを付与する。
    「質問＋回答」を結合してベクトル化することで、
    職員が「〇〇について学生はどう質問している？」と検索したとき
    両方の文脈でヒットさせる。

    Args:
        log_id:      chat_logs.id (uuid文字列)
        user_query:  学生の質問文
        ai_response: AIの回答文（先頭500文字のみ使用）

    Returns:
        True: 成功 / False: 失敗（ログ出力済み）
    """
    # ベクトル化するテキスト: 質問を主軸に回答の要点を添える
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
        logger.error(f"Failed to update chat_log embedding ({log_id}): {e}")
        return False


# ------------------------------------------------------------------
# chat_logs: 一括ベクトル化（管理者用API から呼ぶ）
# ------------------------------------------------------------------

async def bulk_vectorize_chat_logs(limit: int = 200) -> dict:
    """
    embeddingが未設定の chat_logs を最大 `limit` 件ベクトル化する。

    Returns:
        {"processed": int, "succeeded": int, "failed": int}
    """
    try:
        res = db_client.client.table("chat_logs") \
            .select("id, user_query, ai_response") \
            .is_("embedding", "null") \
            .limit(limit) \
            .execute()
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
        # レート制限対策
        await asyncio.sleep(0.3)

    logger.info(f"bulk_vectorize_chat_logs: {succeeded} ok / {failed} failed")
    return {"processed": len(rows), "succeeded": succeeded, "failed": failed}


# ------------------------------------------------------------------
# anonymous_comments: 1件ベクトル化
# ------------------------------------------------------------------

async def vectorize_comment(comment_id: str, comment_text: str) -> bool:
    """
    anonymous_comments の1件にembeddingを付与する。
    """
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
        logger.error(f"Failed to update comment embedding ({comment_id}): {e}")
        return False


# ------------------------------------------------------------------
# anonymous_comments: 一括ベクトル化（管理者用API から呼ぶ）
# ------------------------------------------------------------------

async def bulk_vectorize_comments(limit: int = 200) -> dict:
    """
    embeddingが未設定の anonymous_comments を最大 `limit` 件ベクトル化する。

    Returns:
        {"processed": int, "succeeded": int, "failed": int}
    """
    try:
        res = db_client.client.table("anonymous_comments") \
            .select("id, comment") \
            .is_("embedding", "null") \
            .not_.is_("comment", "null") \
            .limit(limit) \
            .execute()
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