import logging
import asyncio
from core.database import db_client
from models.schemas import ChatLogCreate

logger = logging.getLogger(__name__)

class ChatLogService:
    """
    チャットログの永続化を担当するサービス
    """

    @staticmethod
    async def save_log_async(log_data: ChatLogCreate):
        """
        チャットログをDBに保存する（非同期実行用）
        """
        try:
            # Pydanticモデルを辞書に変換
            data = log_data.model_dump(mode='json')

            # SupabaseへのInsert実行
            # 注意: ベクトル(embedding)はここでは計算せず、まずはテキストログを確実に残すことを優先します
            response = db_client.client.table("chat_logs").insert(data).execute()
            
            if response.data:
                # 開発用ログ: 保存成功
                logger.info(f"Chat log saved. ID: {response.data[0].get('id')}")
            else:
                logger.warning("Chat log saved but no data returned.")

        except Exception as e:
            # ログ保存の失敗がユーザー体験に影響しないよう、エラーはログ出力にとどめる
            logger.error(f"Failed to save chat log: {e}", exc_info=True)