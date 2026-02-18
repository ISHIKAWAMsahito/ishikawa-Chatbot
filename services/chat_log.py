"""
services/chat_log.py

チャットログの永続化を担当するサービス。
ログ保存後、バックグラウンドで embedding を付与することで
stats.html でのベクトル検索を可能にする。
"""
import asyncio
import logging

from core.database import db_client
from models.schemas import ChatLogCreate

logger = logging.getLogger(__name__)


class ChatLogService:
    """
    チャットログの永続化を担当するサービス
    """

    @staticmethod
    async def save_log_async(log_data: ChatLogCreate) -> None:
        """
        チャットログをDBに保存し、保存成功後に
        バックグラウンドタスクとして embedding を付与する。
        """
        saved_id: str | None = None

        try:
            data = log_data.model_dump(mode="json")
            response = db_client.client.table("chat_logs").insert(data).execute()

            if response.data:
                saved_id = response.data[0].get("id")
                logger.info(f"Chat log saved. ID: {saved_id}")
            else:
                logger.warning("Chat log saved but no data returned.")

        except Exception as e:
            # ログ保存失敗はユーザー体験に影響させない
            logger.error(f"Failed to save chat log: {e}", exc_info=True)
            return

        # ログ保存成功後にベクトル化をバックグラウンドで実行
        if saved_id:
            asyncio.create_task(
                _vectorize_saved_log(
                    log_id=saved_id,
                    user_query=log_data.user_query,
                    ai_response=log_data.ai_response,
                )
            )


async def _vectorize_saved_log(
    log_id: str,
    user_query: str,
    ai_response: str,
) -> None:
    """
    保存済みのチャットログに embedding を付与する内部タスク。
    インポートをローカルスコープで行うことで循環インポートを回避する。
    """
    try:
        from services.vectorize_logs import vectorize_chat_log  # 遅延インポート
        await vectorize_chat_log(log_id, user_query, ai_response)
    except Exception as e:
        # ベクトル化失敗はサービス継続に影響させない
        logger.warning(f"Background vectorization failed for log {log_id}: {e}")