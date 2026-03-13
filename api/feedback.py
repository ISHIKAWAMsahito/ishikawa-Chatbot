import asyncio
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from core.dependencies import require_auth, require_auth_client
from core.database import db_client
from services.vectorize_logs import vectorize_comment

router = APIRouter()

class FeedbackRequest(BaseModel):
    feedback_id: str
    rating: str
    comment: str = Field("", max_length=2000)

class FeedbackManager:
    """フィードバック管理クラス(Supabase DB & 自動ベクトル化対応版)"""
    
    async def save_feedback(self, feedback_id: str, rating: str, comment: str = ""):
        try:
            # 1. 評価値の数値化
            rating_val = 5 if rating in ["good", "resolved"] else 1 if rating in ["bad", "not_resolved"] else 0
            
            # 2. 既存のチャットログへの評価 (feedback_idが指定されている場合)
            if feedback_id and feedback_id != "anonymous":
                db_client.client.table("chat_logs").update({
                    "rating": rating_val,
                    "comment": comment
                }).eq("id", feedback_id).execute()
                logging.info(f"チャットログ(ID: {feedback_id})に評価を保存しました")

            # 3. 匿名の意見・要望 (anonymous_commentsへの保存とベクトル化)
            if comment and comment.strip():
                res = db_client.client.table("anonymous_comments").insert({
                    "comment": comment.strip()
                }).execute()
                
                if res.data:
                    new_comment_id = res.data[0]["id"]
                    # 確実にベクトル化が終わるのを待つ
                    await vectorize_comment(new_comment_id, comment.strip())
                    logging.info(f"コメント自動ベクトル化完了(ID: {new_comment_id})")

        except Exception as e:
            logging.error(f"フィードバック保存エラー: {e}")
            raise

    async def get_feedback_stats(self) -> Dict[str, Any]:
        """Supabase DBから統計情報を取得"""
        try:
            res = db_client.client.table("chat_logs").select("rating").neq("rating", None).execute()
            data = res.data or []
            total = len(data)
            resolved = sum(1 for d in data if d.get("rating", 0) >= 4)
            not_resolved = total - resolved
            rate = (resolved / total * 100) if total > 0 else 0
            
            return {
                "total": total,
                "resolved": resolved,
                "not_resolved": not_resolved,
                "rate": round(rate, 1)
            }
        except Exception as e:
            logging.error(f"統計取得エラー: {e}")
            return {"total": 0, "resolved": 0, "not_resolved": 0, "rate": 0}

feedback_manager = FeedbackManager()

@router.post("/")
async def submit_feedback(request: FeedbackRequest, user: dict = Depends(require_auth_client)):
    try:
        await feedback_manager.save_feedback(
            feedback_id=request.feedback_id,
            rating=request.rating,
            comment=request.comment
        )
        return {"status": "success", "message": "フィードバックを受け付けました"}
    except Exception as e:
        logging.error(f"APIエラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="処理に失敗しました。")

@router.get("/stats")
async def get_stats(user: dict = Depends(require_auth)):
    return await feedback_manager.get_feedback_stats()