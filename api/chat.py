from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List

from core.database import get_db
from models.schemas import ChatQuery, FeedbackCreate, FeedbackRead

# ---------------------------------------------------------
# 修正箇所: インポート元を整理
# ---------------------------------------------------------
# utils からインポート
from services.utils import get_or_create_session_id

# chat_logic からインポート
# ※ analyze_feedback_trends が services/chat_logic.py に定義されている必要があります
from services.chat_logic import (
    enhanced_chat_logic, 
    history_manager, 
    analyze_feedback_trends 
)
# ---------------------------------------------------------

router = APIRouter()

@router.post("/chat", summary="AIチャット送信 (ストリーミング)")
async def chat_endpoint(
    request: Request,
    query: ChatQuery,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    ユーザーからの質問を受け付け、RAG + LLM で回答をストリーミング生成します。
    """
    # ジェネレータを StreamingResponse に渡す
    return StreamingResponse(
        enhanced_chat_logic(request, query),
        media_type="text/event-stream"
    )

@router.get("/history", summary="チャット履歴の取得")
def get_history(request: Request):
    """
    現在のセッションの会話履歴を取得します。
    """
    session_id = get_or_create_session_id(request)
    return history_manager.get_history(session_id)

@router.post("/feedback", response_model=FeedbackRead, summary="回答へのフィードバック送信")
def create_feedback(
    feedback: FeedbackCreate,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    AIの回答に対する評価（Good/Bad）とコメントを保存します。
    """
    session_id = get_or_create_session_id(request)
    # 実際の実装に合わせてDB保存処理を記述
    # ここではダミーレスポンスを返します
    return FeedbackRead(
        id=1,
        session_id=session_id,
        rating=feedback.rating,
        comment=feedback.comment,
        created_at="2025-01-01T00:00:00"
    )

@router.get("/analyze", summary="フィードバック分析 (管理者用)")
async def analyze_feedback(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    蓄積されたフィードバックを分析し、改善レポートを生成します。
    """
    # ダミーデータ（実際はDBから取得）
    dummy_logs = [
        {"rating": "bad", "comment": "回答が遅い"},
        {"rating": "good", "comment": "分かりやすかった"}
    ]
    
    return StreamingResponse(
        analyze_feedback_trends(dummy_logs),
        media_type="text/event-stream"
    )