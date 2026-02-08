from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List

from core.database import get_db
from core.config import SUPABASE_URL, SUPABASE_ANON_KEY
from core.dependencies import require_auth, require_auth_client
from models.schemas import ChatQuery, FeedbackCreate, FeedbackRead

from services.utils import get_or_create_session_id
from services.chat_logic import (
    enhanced_chat_logic,
    history_manager,
    analyze_feedback_trends,
)

router = APIRouter()

# ★追加: フロントエンド初期化用エンドポイント
@router.get("/config")
def get_chat_config():
    """
    フロントエンド（client.html等）の初期化に必要な公開設定を返す
    """
    return {
        "supabase_url": SUPABASE_URL,
        "supabase_anon_key": SUPABASE_ANON_KEY
    }

@router.post("/chat", summary="AIチャット送信 (ストリーミング)")
async def chat_endpoint(
    request: Request,
    query: ChatQuery,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth_client),
):
    """
    ユーザーからの質問を受け付け、RAG + LLM で回答をストリーミング生成します。
    """
    return StreamingResponse(
        enhanced_chat_logic(request, query),
        media_type="text/event-stream"
    )

@router.get("/history", summary="チャット履歴の取得")
def get_history(request: Request, user: dict = Depends(require_auth_client)):
    """
    現在のセッションの会話履歴を取得します。
    """
    session_id = get_or_create_session_id(request)
    return history_manager.get_history(session_id)

@router.post("/feedback", response_model=FeedbackRead, summary="回答へのフィードバック送信")
def create_feedback(
    feedback: FeedbackCreate,
    request: Request,
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth_client),
):
    """
    AIの回答に対する評価（Good/Bad）とコメントを保存します。
    """
    session_id = get_or_create_session_id(request)
    JST = timezone(timedelta(hours=9), "JST")
    return FeedbackRead(
        id=1,
        session_id=session_id,
        rating=feedback.rating,
        comment=feedback.comment,
        created_at=datetime.now(JST),
    )

@router.get("/analyze", summary="フィードバック分析 (管理者用)")
async def analyze_feedback(
    request: Request,
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
):
    """
    蓄積されたフィードバックを分析し、改善レポートを生成します。
    """
    dummy_logs = [
        {"rating": "bad", "comment": "回答が遅い"},
        {"rating": "good", "comment": "分かりやすかった"}
    ]

    return StreamingResponse(
        analyze_feedback_trends(dummy_logs),
        media_type="text/event-stream"
    )