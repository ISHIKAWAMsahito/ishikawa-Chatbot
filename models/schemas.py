from typing import Optional, Any
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Chat / RAG Request Models
# -----------------------------------------------------------------------------
class ChatQuery(BaseModel):
    """
    チャットリクエスト用スキーマ
    """
    # ロジック側で .question を参照しているため、フィールド名を合わせます
    question: str = Field(..., description="ユーザーからの質問文", alias="query") 
    
    # 以下はデフォルト値を設定し、必須ではないようにします
    collection: str = Field("default", description="検索対象のコレクション名")
    top_k: int = Field(5, description="検索で取得するドキュメント数")
    embedding_model: str = Field("models/gemini-embedding-001", description="使用するEmbeddingモデル")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "question": "成績証明書の発行方法を教えて",
                "collection": "university_docs",
                "top_k": 5
            }
        }

# -----------------------------------------------------------------------------
# Feedback Models
# -----------------------------------------------------------------------------
class FeedbackCreate(BaseModel):
    """
    フィードバック作成（受信）用スキーマ
    api/chat.py で FeedbackCreate としてインポートされているため、このクラス名が必要です。
    """
    rating: str = Field(..., description="評価 ('good', 'bad' など)")
    comment: Optional[str] = Field(None, description="ユーザーからのコメント")

    class Config:
        json_schema_extra = {
            "example": {
                "rating": "good",
                "comment": "とても分かりやすかったです。"
            }
        }

class FeedbackRead(BaseModel):
    """
    フィードバック返却（表示）用スキーマ
    """
    id: int
    session_id: str
    rating: str
    comment: Optional[str]
    created_at: Any 

    class Config:
        from_attributes = True