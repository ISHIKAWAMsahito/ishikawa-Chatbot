import json
import logging
from typing import List, Optional, Union
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import google.generativeai as genai

from core.database import db_client
from core.dependencies import require_auth
from core.config import GEMINI_API_KEY

# プロンプトのインポート
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
    # 修正: DBがUUIDを返すため str に変更 (intだとエラーになる)
    id: str 
    created_at: str
    rating: Optional[str] = None
    # DBのカラム名 'content' を Pydanticの 'comment' フィールドにマッピング
    comment: Optional[str] = Field(None, alias="content")

    class Config:
        populate_by_name = True

class AnalyzeRequest(BaseModel):
    target_date: Optional[str] = None

# ---------------------------------------------------------
# 1. 統計データ取得 API (GET /data)
# ---------------------------------------------------------
@router.get("/data", response_model=List[FeedbackItem])
async def get_stats_data(
    user: dict = Depends(require_auth)
):
    try:
        response = db_client.client.table("anonymous_comments") \
            .select("*") \
            .order("created_at", desc=True) \
            .limit(100) \
            .execute()
        
        return response.data

    except Exception as e:
        logger.error(f"Error fetching stats data: {e}", exc_info=True)
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
        # DBから分析対象データを取得 (contentのみ取得)
        db_res = db_client.client.table("anonymous_comments") \
            .select("created_at, content") \
            .order("created_at", desc=True) \
            .limit(50) \
            .execute()
        
        # 内部関数としてジェネレータを定義
        async def stream_generator():
            if not db_res.data:
                yield "data: " + json.dumps({"content": "分析データがありません。"}) + "\n\n"
                return

            data_summary = json.dumps(db_res.data, ensure_ascii=False, indent=2)
            prompt = FEEDBACK_ANALYSIS.format(summary=data_summary)

            model = genai.GenerativeModel(MODEL_NAME)
            response_stream = model.generate_content(prompt, stream=True)

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