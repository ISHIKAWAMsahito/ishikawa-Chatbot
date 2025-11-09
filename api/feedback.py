import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException
from models.schemas import FeedbackRequest, AIResponseFeedbackRequest
from services.feedback import feedback_manager
# ↓↓↓ [修正] モジュール本体をインポート
from core import database
# ↑↑↑ [修正]
from core.config import JST

router = APIRouter()

@router.post("/feedback")
async def save_feedback(feedback: FeedbackRequest):
    """フィードバックを保存"""
    try:
        feedback_manager.save_feedback(feedback.feedback_id, feedback.rating, feedback.comment)
        return {"message": "フィードバックを保存しました"}
    except Exception as e:
        logging.error(f"フィードバック保存エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback/ai-response")
async def save_ai_response_feedback(feedback: AIResponseFeedbackRequest):
    """AI回答へのフィードバックを保存"""
    try:
        # ↓↓↓ [修正] "database." を追加
        database.db_client.client.table("anonymous_comments").insert({
            "comment": f"Q: {feedback.user_question}\nA: {feedback.ai_response}",
            "created_at": datetime.now(JST).isoformat(),
            "rating": feedback.rating
        }).execute()
        # ↑↑↑ [修正]
        return {"message": "AI回答へのフィードバックを保存しました"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/feedback/stats")
async def get_feedback_stats():
    """フィードバック統計を取得"""
    try:
        stats = feedback_manager.get_feedback_stats()
        return stats
    except Exception as e:
        logging.error(f"フィードバック統計取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))