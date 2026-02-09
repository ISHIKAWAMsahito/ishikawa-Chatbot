import logging
import json
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
    embedding: Optional[List[float]] = None 

# --- Helper Functions ---
def generate_embedding(text: str) -> List[float]:
    """Geminiでテキストをベクトル化する"""
    try:
        clean_text = text.replace("\n", " ")
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=clean_text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail="エンベディング生成に失敗しました")

def process_db_item(item: dict) -> dict:
    """
    DBからの生データをAPIレスポンス用に加工する
    1. category_name -> category
    2. embedding (str) -> embedding (list)
    """
    # 1. カラム名の変更
    if 'category_name' in item:
        item['category'] = item.pop('category_name')
    # categoryもcategory_nameもない場合はデフォルト値を設定
    if 'category' not in item:
        item['category'] = 'general'

    # 2. Embedding文字列のパース ("[-0.1, ...]" -> [-0.1, ...])
    if item.get('embedding') and isinstance(item['embedding'], str):
        try:
            item['embedding'] = json.loads(item['embedding'])
        except Exception as e:
            logger.warning(f"Failed to parse embedding for id {item.get('id')}: {e}")
            item['embedding'] = None
            
    return item

# --- Endpoints ---

@router.get("/", response_model=List[FallbackResponse])
async def list_fallbacks(user: dict = Depends(require_auth)):
    """登録されているQ&Aリストを取得"""
    try:
        response = db_client.client.table("category_fallbacks") \
            .select("id, category_name, question, answer, created_at, embedding") \
            .order("category_name", desc=False) \
            .order("created_at", desc=True) \
            .execute()
        
        # データを加工して返す
        return [process_db_item(item) for item in response.data]

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
        
        data = {
            "category_name": fallback.category, 
            "question": fallback.question,
            "answer": fallback.answer,
            "embedding": embedding
        }
        
        response = db_client.client.table("category_fallbacks").insert(data).execute()
        
        if not response.data:
            raise HTTPException(status_code=500, detail="保存に失敗しました")
        
        return process_db_item(response.data[0])

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
        
        return process_db_item(response.data[0])

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

@router.post("/vectorize-all")
async def vectorize_all_fallbacks(user: dict = Depends(require_auth)):
    """
    embeddingが空、またはすべてのQ&Aに対してベクトル化を実行して更新する
    """
    try:
        response = db_client.client.table("category_fallbacks").select("*").execute()
        all_records = response.data
        
        updated_count = 0
        
        for record in all_records:
            # embeddingが無い場合に実行
            if not record.get('embedding') and record.get('question'):
                try:
                    new_embedding = generate_embedding(record['question'])
                    
                    db_client.client.table("category_fallbacks") \
                        .update({"embedding": new_embedding}) \
                        .eq("id", record['id']) \
                        .execute()
                        
                    updated_count += 1
                except Exception as emb_err:
                    logger.error(f"Failed to vectorize ID {record['id']}: {emb_err}")
                    continue

        return {"message": f"{updated_count}件のベクトル化が完了しました。"}

    except Exception as e:
        logger.error(f"Error vectorizing all: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"一括処理中にエラーが発生しました: {str(e)}")