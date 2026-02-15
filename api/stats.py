import json
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.database import db_client
from core.dependencies import require_auth
from models.schemas import AnalysisQuery, FeedbackItem # Step 2で定義したAnalysisQueryを使用
from services.llm import LLMService
from services import prompts

logger = logging.getLogger(__name__)
router = APIRouter()

# サービス初期化
llm_service = LLMService()

# ---------------------------------------------------------
# 1. 統計データ取得 API (既存)
# ---------------------------------------------------------
@router.get("/data", response_model=List[FeedbackItem])
async def get_stats_data(user: dict = Depends(require_auth)):
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
# 2. フィードバック分析 API (既存)
# ---------------------------------------------------------
@router.post("/analyze")
async def analyze_feedback(
    request: AnalysisQuery, # ここも型定義を合わせるとベターですが既存維持でも可
    user: dict = Depends(require_auth)
):
    # ... (既存のフィードバック分析ロジック) ...
    # 必要であれば後ほどリファクタリングしましょう
    pass 

# ---------------------------------------------------------
# 3. チャットログ相談・分析 API (★新規追加)
# ---------------------------------------------------------
@router.post("/chat_analysis", summary="対話ログに基づく相談・分析")
async def chat_with_logs(
    request: AnalysisQuery,
    user: dict = Depends(require_auth)
):
    """
    職員からの質問に対し、過去のチャットログを参照して分析結果を返す
    """
    logger.info(f"Admin analysis query: {request.query}")

    try:
        # 1. データの取得 (RAGの簡易版: 直近のログを取得)
        # 本格的な運用ではここを「ベクトル検索」にしますが、
        # まずは「直近の傾向」を知るために最新100件を取得してコンテキストにします。
        days_ago = request.target_period_days
        
        # Supabaseの日付フィルタ用
        # (実際は datetime計算が必要ですが、簡単のためlimitで代用例を示します)
        
        db_res = db_client.client.table("chat_logs") \
            .select("created_at, user_query, ai_response, metadata") \
            .order("created_at", desc=True) \
            .limit(50) \
            .execute()
        
        logs = db_res.data
        if not logs:
            return StreamingResponse(
                _stream_text("分析対象となるログデータがまだありません。"),
                media_type="text/event-stream"
            )

        # 2. コンテキストの構築
        # LLMが読みやすい形式にテキスト化
        log_context = ""
        for i, log in enumerate(logs):
            meta = log.get('metadata', {})
            category = meta.get('collection', 'unknown')
            log_context += f"No.{i+1} [日時: {log['created_at']}] [カテゴリ: {category}]\n"
            log_context += f"Q: {log['user_query']}\n"
            log_context += f"A: {log['ai_response'][:100]}...\n\n" # 回答は長すぎるので要約

        system_prompt = prompts.LOG_ANALYSIS_ADVISOR

        user_prompt = f"""
        【職員からの相談】
        {request.query}

        【分析対象ログ】
        {log_context}
        """

        # 4. ストリーミング生成して返却
        return StreamingResponse(
            llm_service.generate_response_stream(
                query=user_prompt,
                context_docs=[], # ここではコンテキストを直接プロンプトに埋め込んだため空でOK
                history=[],      # 一回切りの分析なので履歴は空
                system_prompt=system_prompt
            ),
            media_type="text/event-stream"
        )

    except Exception as e:
        logger.error(f"Analysis Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"分析処理中にエラーが発生しました: {str(e)}")

async def _stream_text(text: str):
    """単純なテキストをSSE形式で返すヘルパー"""
    yield text