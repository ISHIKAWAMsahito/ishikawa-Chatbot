import json
import logging
from typing import List, Optional, Any, Dict
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import google.generativeai as genai

from core.database import db_client
from core.security import require_auth
from core.config import GEMINI_API_KEY
# prompts.py がある場合はインポート、なければ直接記述
try:
    from core.prompts import FEEDBACK_ANALYSIS
except ImportError:
    FEEDBACK_ANALYSIS = "以下のログを分析してください:\n{summary}"

logger = logging.getLogger(__name__)
router = APIRouter()

# Gemini設定
genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash"

# --- Pydantic Models ---
class FeedbackItem(BaseModel):
    id: int  # または str (DB定義に合わせる)
    created_at: str
    rating: Optional[str] = None
    comment: Optional[str] = None
    # log: Optional[Dict[str, Any]] = None # 必要に応じてコメントアウト解除

class AnalyzeRequest(BaseModel):
    target_date: Optional[str] = None

# ---------------------------------------------------------
# 1. 統計データ取得 API (GET /data)
# ---------------------------------------------------------
@router.get("/data", response_model=List[FeedbackItem])
async def get_stats_data(
    user: dict = Depends(require_auth)  # 管理者権限必須
):
    """
    フィードバック履歴を取得する (stats.html用)
    """
    try:
        # Supabaseのテーブル名 'feedback' を想定
        # ※もしテーブル名が異なる場合(chat_logsなど)はここを修正してください
        response = db_client.client.table("feedback") \
            .select("*") \
            .order("created_at", desc=True) \
            .limit(100) \
            .execute()
        
        return response.data

    except Exception as e:
        logger.error(f"Error fetching stats data: {e}", exc_info=True)
        # テーブルが存在しない等のエラー詳細をログに出し、500エラーを返す
        raise HTTPException(status_code=500, detail="データの取得に失敗しました")

# ---------------------------------------------------------
# 2. AI分析 API (POST /analyze)
# ---------------------------------------------------------
@router.post("/analyze")
async def analyze_feedback(
    request: AnalyzeRequest,
    user: dict = Depends(require_auth)
):
    """
    直近のログをGeminiに分析させ、ストリーミングで返す
    """
    try:
        # 分析データの取得
        db_res = db_client.client.table("feedback") \
            .select("created_at, rating, comment") \
            .order("created_at", desc=True) \
            .limit(50) \
            .execute()
        
        if not db_res.data:
            yield "data: " + json.dumps({"content": "分析データがありません。"}) + "\n\n"
            return

        data_summary = json.dumps(db_res.data, ensure_ascii=False, indent=2)
        prompt = FEEDBACK_ANALYSIS.format(summary=data_summary)

        model = genai.GenerativeModel(MODEL_NAME)
        response_stream = model.generate_content(prompt, stream=True)

        async def stream_generator():
            for chunk in response_stream:
                if chunk.text:
                    payload = json.dumps({"content": chunk.text})
                    yield f"data: {payload}\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream"
        )

    except Exception as e:
        logger.error(f"Error in AI analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="AI分析エラー")