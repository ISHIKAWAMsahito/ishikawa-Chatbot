import logging
import asyncio
import re
from typing import List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import google.generativeai as genai

from core.dependencies import require_auth
from core import database
from core import settings as app_settings  # core.settings と競合しないようエイリアス
from core.config import GEMINI_API_KEY, EMBEDDING_MODEL
from models.schemas import FallbackCreate, FallbackUpdate

router = APIRouter()

# ---------------------------------------------------------
# レスポンスモデル定義 (Strict Typing / Dict禁止)
# ---------------------------------------------------------
class FallbackResponse(BaseModel):
    id: int
    question: str
    answer: str
    category_name: Optional[str] = None
    embedding: Optional[List[float]] = None
    created_at: Optional[str] = None

class FallbackListResponse(BaseModel):
    fallbacks: List[FallbackResponse]

class MessageResponse(BaseModel):
    message: str
    fallback: Optional[FallbackResponse] = None

# APIキーの設定
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------
# エンドポイント
# ---------------------------------------------------------

# 修正: main.py で prefix="/api/admin/fallbacks" としているため、
# ここを "" (空文字) にすることで /api/admin/fallbacks (末尾スラッシュなし) に対応
@router.get("", response_model=FallbackListResponse)
async def get_all_fallbacks(user: Any = Depends(require_auth)):
    """Q&A(フォールバック)をすべて取得"""
    if not database.db_client:
        raise HTTPException(status_code=503, detail="DB not initialized")
    
    try:
        # 必要なカラムを明示的に指定
        response = database.db_client.client.table("category_fallbacks")\
            .select("id, question, answer, category_name, embedding, created_at")\
            .order("id", desc=True)\
            .execute()
        
        data = response.data if response.data else []
        return FallbackListResponse(fallbacks=data)
    
    except Exception as e:
        logging.error(f"Q&A一覧取得エラー: {e}", exc_info=True)
        # 指針: 詳細なエラー内容はクライアントに返さない
        raise HTTPException(status_code=500, detail="処理に失敗しました。")

@router.post("", response_model=MessageResponse)
async def create_fallback(request: FallbackCreate, user: Any = Depends(require_auth)):
    """新しいQ&Aを作成"""
    if not database.db_client:
        raise HTTPException(status_code=503, detail="DB not initialized")

    try:
        question_text = request.question.strip()
        answer_text = request.answer.strip()
        category_name = request.category_name.strip() if request.category_name else "General"

        embedding = None
        try:
            # 指針: モデル固定 (gemini-embedding-001)
            embedding_response = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=question_text
            )
            embedding = embedding_response["embedding"]
        except Exception as e:
            logging.error(f"ベクトル生成エラー: {e}")
            # ベクトル生成失敗でも登録は許可する場合 (要件によるが今回はログのみ)

        insert_data = {
            "question": question_text,
            "answer": answer_text,
            "category_name": category_name,
            "embedding": embedding
        }
        
        result = database.db_client.client.table("category_fallbacks").insert(insert_data).execute()
        created_item = result.data[0] if result.data else None
        
        return MessageResponse(message="保存しました", fallback=created_item)

    except Exception as e:
        logging.error(f"作成エラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="処理に失敗しました。")

@router.put("/{qa_id}", response_model=MessageResponse)
async def update_fallback(qa_id: int, request: FallbackUpdate, user: Any = Depends(require_auth)):
    """Q&Aを更新"""
    if not database.db_client:
        raise HTTPException(status_code=503, detail="DB not initialized")

    # exclude_unset=True で送信されたフィールドのみ更新
    update_data = request.model_dump(exclude_unset=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="更新するフィールドがありません")

    try:
        # 質問文が変更された場合のみ再埋め込み
        if "question" in update_data:
            new_question = (update_data["question"] or "").strip()
            if not new_question:
                raise HTTPException(status_code=400, detail="Question cannot be empty")
            
            update_data["question"] = new_question
            
            try:
                embedding_response = genai.embed_content(
                    model=EMBEDDING_MODEL,
                    content=new_question
                )
                update_data["embedding"] = embedding_response["embedding"]
            except Exception as e:
                logging.error(f"ベクトル更新エラー: {e}")
                update_data["embedding"] = None

        result = database.db_client.client.table("category_fallbacks")\
            .update(update_data).eq("id", qa_id).execute()
        
        updated_item = result.data[0] if result.data else None
        return MessageResponse(message="更新しました", fallback=updated_item)

    except Exception as e:
        logging.error(f"更新エラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="処理に失敗しました。")

@router.delete("/{qa_id}", response_model=MessageResponse)
async def delete_fallback(qa_id: int, user: Any = Depends(require_auth)):
    """Q&Aを削除"""
    if not database.db_client:
        raise HTTPException(status_code=503, detail="DB not initialized")
    
    try:
        database.db_client.client.table("category_fallbacks").delete().eq("id", qa_id).execute()
        return MessageResponse(message="削除しました")
    except Exception as e:
        logging.error(f"削除エラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="処理に失敗しました。")

@router.post("/vectorize-all", response_model=MessageResponse)
async def vectorize_all_missing_fallbacks(user: Any = Depends(require_auth)):
    """全Q&Aのベクトル修復"""
    if not database.db_client:
        raise HTTPException(status_code=503, detail="DB not initialized")

    logging.info("全Q&Aのベクトル修復処理を開始...")
    try:
        response = database.db_client.client.table("category_fallbacks").select("id, question").execute()
        if not response.data:
            return MessageResponse(message="データがありません。")

        count = 0
        for item in response.data:
            item_id = item['id']
            original_text = item.get('question', '')
            if not original_text: continue
            
            text_to_vectorize = re.sub(r'[\r\n\t]', ' ', original_text).strip()
            if not text_to_vectorize: continue

            try:
                embedding_response = genai.embed_content(
                    model=EMBEDDING_MODEL,
                    content=text_to_vectorize
                )
                database.db_client.client.table("category_fallbacks").update({
                    "embedding": embedding_response["embedding"]
                }).eq("id", item_id).execute()
                
                count += 1
                await asyncio.sleep(1) # API制限回避
            except Exception as e:
                if "429" in str(e): await asyncio.sleep(30)
                logging.error(f"ID {item_id} エラー: {e}")

        return MessageResponse(message=f"修復完了。{count}件のベクトルを再生成しました。")
    except Exception as e:
        logging.error(f"ベクトル修復処理エラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="処理に失敗しました。")