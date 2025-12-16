import os
import json
import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException  # 追加
from pydantic import BaseModel  # 追加
from core.config import BASE_DIR, JST

# ==========================================
# 1. ルーターの定義 (これが不足していました)
# ==========================================
router = APIRouter()

# ユーザーからの評価（Good/Bad/コメント）を保存します。

# シンプル保存: 他の機能がSupabaseを使っているのに対し、このファイルは json ファイルへの読み書きを行っているようです（feedback.json）。

# 集計: 解決率（Resolved rate）などの簡易的な統計情報を返します。

# ==========================================
# 2. データ受信用の型定義 (Pydanticモデル)
# ==========================================
class FeedbackRequest(BaseModel):
    feedback_id: str
    rating: str
    comment: str = ""

# ==========================================
# 3. ロジッククラス (元のコード)
# ==========================================
class FeedbackManager:
    """フィードバック管理クラス(簡素化版)"""
    def __init__(self):
        self.feedback_file = os.path.join(BASE_DIR, "feedback.json")

    def save_feedback(self, feedback_id: str, rating: str, comment: str = ""):
        """フィードバックを保存"""
        try:
            feedback_data = []
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    feedback_data = json.load(f)
            
            feedback_entry = {
                "id": feedback_id,
                "rating": rating,
                "comment": comment,
                "timestamp": datetime.now(JST).isoformat()
            }
            feedback_data.append(feedback_entry)
            
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(feedback_data, f, ensure_ascii=False, indent=2)
            
            logging.info(f"フィードバック保存完了: {feedback_id} - {rating}")
        except Exception as e:
            logging.error(f"フィードバック保存エラー: {e}")
            raise  # エラーを呼び出し元に伝える

    def get_feedback_stats(self) -> Dict[str, Any]:
        """フィードバック統計を取得"""
        try:
            if not os.path.exists(self.feedback_file):
                return {"total": 0, "resolved": 0, "not_resolved": 0, "rate": 0}
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
            total = len(feedback_data)
            resolved = sum(1 for fb in feedback_data if fb['rating'] == 'resolved')
            not_resolved = total - resolved
            rate = (resolved / total * 100) if total > 0 else 0
            return {
                "total": total,
                "resolved": resolved,
                "not_resolved": not_resolved,
                "rate": round(rate, 1)
            }
        except Exception as e:
            logging.error(f"フィードバック統計取得エラー: {e}")
            return {"total": 0, "resolved": 0, "not_resolved": 0, "rate": 0}

# グローバルインスタンス
feedback_manager = FeedbackManager()

# ==========================================
# 4. APIエンドポイントの実装 (ここが重要)
# ==========================================
@router.post("/")
async def submit_feedback(request: FeedbackRequest):
    """
    フィードバックを受け取るAPIエンドポイント
    URL: /api/client/feedback/ (main.pyの設定による)
    """
    try:
        feedback_manager.save_feedback(
            feedback_id=request.feedback_id,
            rating=request.rating,
            comment=request.comment
        )
        return {"status": "success", "message": "フィードバックを受け付けました"}
    except Exception as e:
        logging.error(f"APIエラー: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/stats")
async def get_stats():
    """
    統計情報を取得するAPIエンドポイント (管理者用などで使用可能)
    """
    return feedback_manager.get_feedback_stats()