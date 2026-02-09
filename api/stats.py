import json
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import google.generativeai as genai

from core.database import db_client
from core.security import require_auth
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
    id: int
    created_at: str
    rating: Optional[str] = None
    comment: Optional[str] = None

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
        # テーブル名は環境に合わせて 'feedback' か 'chat_logs' 等に調整してください
        response = db_client.client.table("feedback") \
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
        # 1. DBから分析対象データを取得
        db_res = db_client.client.table("feedback") \
            .select("created_at, rating, comment") \
            .order("created_at", desc=True) \
            .limit(50) \
            .execute()
        
        # 2. 内部関数としてジェネレータを定義 (ここが修正ポイント)
        #    メイン関数内で yield は使わず、この内部関数内でのみ yield を行う
        async def stream_generator():
            if not db_res.data:
                # データがない場合もストリーム形式でメッセージを返す
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

        # 3. ジェネレータを渡してレスポンスを返す (yieldではなくreturnする)
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream"
        )

    except Exception as e:
        logger.error(f"Error in AI analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="AI分析エラー")