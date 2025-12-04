from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from core.config import ACTIVE_COLLECTION_NAME

class ChatQuery(BaseModel):
    query: str
    model: str = "gemini-2.5-flash"
    embedding_model: str = "models/embedding-001"
    top_k: int = 5
    collection: str = ACTIVE_COLLECTION_NAME

class ClientChatQuery(BaseModel):
    query: str

class FeedbackRequest(BaseModel):
    feedback_id: str
    rating: str
    comment: str = ""

class AIResponseFeedbackRequest(BaseModel):
    user_question: str
    ai_response: str
    rating: str

class Settings(BaseModel):
    model: Optional[str] = None
    collection: Optional[str] = None
    embedding_model: Optional[str] = None
    top_k: Optional[int] = None