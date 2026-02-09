import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

from core.database import db_client
from core.dependencies import require_auth
from core.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)
router = APIRouter()

# Gemini Embeddings 設定
genai.configure(api_key=GEMINI_API_KEY)
EMBEDDING_MODEL = "models/gemini-embedding-001"

# --- Pydantic Models ---
# フロントエンド(API)側は "category" という名前で扱います
class FallbackBase(BaseModel):
    category: str
    question: str
    answer: str

class FallbackCreate(FallbackBase):
    pass

class FallbackUpdate(BaseModel):
    category: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None

class FallbackResponse(FallbackBase):
    id: int
    created_at: str

# --- Helper Functions ---
def generate_embedding(text: str) -> List[float]:
    """Geminiでテキストをベクトル化する"""
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail="エンベディング生成に失敗しました")

# --- Endpoints ---

@router.get("/", response_model=List[FallbackResponse])
async def list_fallbacks(user: dict = Depends(require_auth)):
    """登録されているQ&Aリストを取得"""
    try:
        # DBのカラム名 'category_name' を指定して取得
        response = db_client.client.table("category_fallbacks") \
            .select("id, category_name, question, answer, created_at") \
            .order("category_name", desc=False) \
            .order("created_at", desc=True) \
            .execute()
        
        # DBの 'category_name' を API仕様の 'category' に変換してリスト化
        data = []
        for item in response.data:
            # category_name があれば取り出して category に付け替える
            # (万が一 null の場合は 'general' 等にする)
            cat_val = item.pop('category_name', 'general')
            item['category'] = cat_val
            data.append(item)
            
        return data
    except Exception as e:
        logger.error(f"Error fetching fallbacks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"データの取得に失敗しました: {str(e)}")

@router.post("/", response_model=FallbackResponse)
async def create_fallback(
    fallback: FallbackCreate, 
    user: dict = Depends(require_auth)
):
    """新しいQ&Aを作成"""
    try:
        embedding = generate_embedding(fallback.question)
        
        # 保存時は DBカラム名 'category_name' をキーにする
        data = {
            "category_name": fallback.category, 
            "question": fallback.question,
            "answer": fallback.answer,
            "embedding": embedding
        }
        
        response = db_client.client.table("category_fallbacks").insert(data).execute()
        
        if not response.data:
            raise HTTPException(status_code=500, detail="保存に失敗しました")
        
        # レスポンス用に変換 (DBから返ってきたデータも category_name なので変換が必要)
        result_item = response.data[0]
        result_item['category'] = result_item.pop('category_name', fallback.category)
            
        return result_item
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating fallback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{fallback_id}", response_model=FallbackResponse)
async def update_fallback(
    fallback_id: int, 
    fallback: FallbackUpdate, 
    user: dict = Depends(require_auth)
):
    """Q&Aを更新"""
    try:
        update_data = {}
        
        # 入力 'category' -> DB 'category_name'
        if fallback.category:
            update_data["category_name"] = fallback.category
        
        if fallback.answer:
            update_data["answer"] = fallback.answer
        
        if fallback.question:
            update_data["question"] = fallback.question
            update_data["embedding"] = generate_embedding(fallback.question)
            
        if not update_data:
            raise HTTPException(status_code=400, detail="更新データがありません")

        response = db_client.client.table("category_fallbacks") \
            .update(update_data) \
            .eq("id", fallback_id) \
            .execute()
            
        if not response.data:
            raise HTTPException(status_code=404, detail="対象が見つかりません")
        
        # レスポンス用に変換
        result_item = response.data[0]
        if 'category_name' in result_item:
            result_item['category'] = result_item.pop('category_name')
            
        return result_item
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating fallback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{fallback_id}")
async def delete_fallback(
    fallback_id: int, 
    user: dict = Depends(require_auth)
):
    """Q&Aを削除"""
    try:
        response = db_client.client.table("category_fallbacks") \
            .delete() \
            .eq("id", fallback_id) \
            .execute()
            
        return {"status": "success", "id": fallback_id}
    except Exception as e:
        logger.error(f"Error deleting fallback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="削除に失敗しました")