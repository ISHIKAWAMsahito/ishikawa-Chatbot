from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Chat / RAG Request Models
# -----------------------------------------------------------------------------
class ChatQuery(BaseModel):
    """
    チャットリクエスト用スキーマ（DoS対策: question に長さ制限）
    """
    question: str = Field(
        ...,
        description="ユーザーからの質問文",
        alias="query",
        min_length=1,
        max_length=4000,
    )
    collection: str = Field("default", description="検索対象のコレクション名")
    top_k: int = Field(5, description="検索で取得するドキュメント数")
    embedding_model: str = Field(
        "models/gemini-embedding-001", description="使用するEmbeddingモデル"
    )
    search_mode: Literal["hybrid", "documents", "faq"] = Field(
        "hybrid",
        description=(
            "検索モード: "
            "hybrid=資料+FAQを同時検索（デフォルト）, "
            "documents=一般資料のみ, "
            "faq=FAQのみ"
        ),
    )

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "question": "成績証明書の発行方法を教えて",
                "collection": "university_docs",
                "top_k": 5,
                "search_mode": "hybrid",
            }
        }


# -----------------------------------------------------------------------------
# Feedback Models
# -----------------------------------------------------------------------------
class FeedbackCreate(BaseModel):
    """フィードバック作成（受信）用スキーマ"""
    rating: str = Field(..., description="評価 ('good', 'bad' など)")
    comment: Optional[str] = Field(
        None, max_length=2000, description="ユーザーからのコメント（DoS対策）"
    )

    class Config:
        json_schema_extra = {
            "example": {"rating": "good", "comment": "とても分かりやすかったです。"}
        }


class FeedbackRead(BaseModel):
    """フィードバック返却（表示）用スキーマ"""
    id: int
    session_id: str
    rating: str
    comment: Optional[str]
    created_at: datetime = Field(..., description="作成日時")

    class Config:
        from_attributes = True


# -----------------------------------------------------------------------------
# Fallbacks (Q&A)
# -----------------------------------------------------------------------------
class FallbackCreate(BaseModel):
    """フォールバック Q&A 作成用（DoS対策: max_length 付与）"""
    question: str = Field(..., min_length=1, max_length=2000)
    answer: str = Field(..., min_length=1, max_length=10000)
    category_name: str = Field(..., min_length=1, max_length=256)


class FallbackUpdate(BaseModel):
    """フォールバック Q&A 更新用（部分更新・任意フィールド）"""
    question: Optional[str] = Field(None, min_length=1, max_length=2000)
    answer: Optional[str] = Field(None, min_length=1, max_length=10000)
    category_name: Optional[str] = Field(None, min_length=1, max_length=256)


# -----------------------------------------------------------------------------
# System / Collections
# -----------------------------------------------------------------------------
class CreateCollectionRequest(BaseModel):
    """コレクション作成リクエスト"""
    name: Optional[str] = Field(None, max_length=256, description="コレクション名（将来用）")


# -----------------------------------------------------------------------------
# System Settings Models
# -----------------------------------------------------------------------------
class Settings(BaseModel):
    """システム設定変更用スキーマ"""
    model: Optional[str] = None
    collection: Optional[str] = None
    embedding_model: Optional[str] = None
    top_k: Optional[int] = None


# -----------------------------------------------------------------------------
# Chat Logs / Analysis Models
# -----------------------------------------------------------------------------
class FeedbackItem(BaseModel):
    """管理者画面の統計データ一覧で表示するためのフィードバック項目スキーマ"""
    id: str
    created_at: datetime = Field(..., description="作成日時")
    rating: Optional[str] = None
    comment: Optional[str] = Field(None, alias="content")

    class Config:
        populate_by_name = True
        from_attributes = True


class ChatLogCreate(BaseModel):
    """チャットログ保存用スキーマ"""
    session_id: str = Field(..., description="クライアントのセッションID")
    user_query: str = Field(..., description="学生の質問内容")
    ai_response: str = Field(..., description="AIの回答内容")
    metadata: Optional[dict] = Field(
        default_factory=dict, description="参照元ドキュメント情報など"
    )


class ChatLogRead(BaseModel):
    """チャットログ読み出し用スキーマ（管理者分析用）"""
    id: str
    session_id: str
    user_query: str
    ai_response: str
    created_at: datetime = Field(..., description="作成日時")

    class Config:
        from_attributes = True


class AnalysisQuery(BaseModel):
    """管理者からの分析依頼リクエスト"""
    query: str = Field(..., min_length=1, max_length=1000, description="分析官への質問")
    target_period_days: int = Field(30, description="分析対象とする過去の日数")


# -----------------------------------------------------------------------------
# Search Result Models
# -----------------------------------------------------------------------------
class DocumentMetadata(BaseModel):
    """検索結果のメタデータ構造定義"""
    source: str = "不明な資料"
    page: Optional[int] = None
    chunk: Optional[int] = None
    file_path: Optional[str] = None
    url: Optional[str] = None
    category: Optional[str] = None
    rerank_reason: Optional[str] = None


class SearchResult(BaseModel):
    """検索結果の共通構造定義"""
    id: str
    content: str
    metadata: DocumentMetadata
    similarity: float = 0.0
    score: float = 0.0