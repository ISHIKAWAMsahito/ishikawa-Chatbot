import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

from core.database import db_client
# ※実際の認証関数のインポートパスに合わせて調整してください
# from api.auth import require_auth_client, require_auth

router = APIRouter()
logger = logging.getLogger(__name__)

# -----------------------------------------
# 1. Pydantic モデルの定義 (ルール 8.2 準拠)
# -----------------------------------------
class FeedbackSubmit(BaseModel):
    chat_log_id: str  # どの会話に対する評価か
    rating: int       # 1(低) - 5(高)
    comment: Optional[str] = ""

# -----------------------------------------
# 2. フィードバック管理クラス (Supabase連携版)
# -----------------------------------------
class FeedbackManager:
    async def save_feedback(self, data: FeedbackSubmit):
        """フィードバックをSupabaseのchat_logsに保存（更新）"""
        try:
            # 既存のチャットログに対して、ratingとcommentを更新する
            response = db_client.client.table("chat_logs").update({
                "rating": data.rating,
                "comment": data.comment
            }).eq("id", data.chat_log_id).execute()
            
            logger.info(f"フィードバック保存完了: {data.chat_log_id} - Rating: {data.rating}")
            return response.data
        except Exception as e:
            logger.error(f"フィードバック保存エラー: {e}")
            raise HTTPException(status_code=500, detail="フィードバックの保存に失敗しました")

    async def get_feedback_stats(self) -> Dict[str, Any]:
        """フィードバック統計を取得（評価なしを除外）"""
        try:
            # Supabaseから評価(rating)が入っているデータのみを取得
            # rating が null ではない（評価済み）データを抽出
            response = db_client.client.table("chat_logs").select("rating").not_.is_("rating", "null").execute()
            feedback_data = response.data
            
            # 1. 評価が行われた総数
            rated_total = len(feedback_data)
            
            if rated_total == 0:
                return {"total": 0, "resolved": 0, "not_resolved": 0, "rate": 0}
            
            # 2. 解決 / 未解決 の仕分け (1-5の数値に基づく)
            # 例: 4と5を「解決」、1, 2, 3を「未解決」とする
            resolved = sum(1 for fb in feedback_data if fb.get("rating", 0) >= 4)
            not_resolved = rated_total - resolved
            
            # 3. 解決率の計算
            rate = (resolved / rated_total * 100)
            
            # （参考）全体のチャットログ数を知りたい場合は別途カウントクエリが必要です
            
            return {
                "rated_total": rated_total, # 評価済みの総件数
                "resolved": resolved,
                "not_resolved": not_resolved,
                "rate": round(rate, 1)
            }
        except Exception as e:
            logger.error(f"フィードバック統計取得エラー: {e}")
            return {"rated_total": 0, "resolved": 0, "not_resolved": 0, "rate": 0}

feedback_manager = FeedbackManager()

# -----------------------------------------
# 3. ルーター定義 (ルール 8.1 準拠)
# -----------------------------------------
# 学生からのフィードバック送信 (認証必須)
# ※ Depends(require_auth_client) は実際のプロジェクトの関数名に合わせてください
@router.post("/submit")
async def submit_feedback(
    feedback_data: FeedbackSubmit, 
    # user: dict = Depends(require_auth_client) 
):
    return await feedback_manager.save_feedback(feedback_data)

# 管理者向けの統計取得 (認証必須)
@router.get("/stats")
async def get_stats(
    # admin: dict = Depends(require_auth)
):
    return await feedback_manager.get_feedback_stats()