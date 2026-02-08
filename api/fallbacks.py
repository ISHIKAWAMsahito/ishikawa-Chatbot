import logging
import asyncio
import re
from fastapi import APIRouter, HTTPException, Depends
import google.generativeai as genai
from core.dependencies import require_auth
from core import database
from core import settings
from core.config import GEMINI_API_KEY
from models.schemas import FallbackCreate, FallbackUpdate

router = APIRouter()

# RAG検索がうまくいかない場合や、頻出質問に対応するための固定Q&A（フォールバック）を管理します。

# CRUD操作: Q&Aの作成、読み取り、更新、削除を行います。

# 自動ベクトル化: 質問文（Question）が登録・更新されると、即座にEmbedding APIを叩いてベクトル化し、類似度検索にかかるようにしています。

# 修復機能: /vectorize-all エンドポイントにより、何らかの理由でベクトルが欠落したQ&Aデータを一括で再計算するメンテナンス機能を備えています。

# APIキーの設定
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# 修正: /api/fallbacks -> /
@router.get("/")
async def get_all_fallbacks(user: dict = Depends(require_auth)):
    """Q&A(フォールバック)をすべて取得"""
    if not database.db_client:
        raise HTTPException(503, "DB not initialized")
    try:
        result = database.db_client.client.table("category_fallbacks").select("id, question, answer, category_name, embedding").order("id", desc=True).execute()
        return {"fallbacks": result.data or []}
    except Exception as e:
        logging.error(f"Q&A一覧取得エラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="処理に失敗しました。")

# 修正: /api/fallbacks -> / （リクエストは Pydantic 必須・dict 禁止）
@router.post("/")
async def create_fallback(request: FallbackCreate, user: dict = Depends(require_auth)):
    """新しいQ&Aを作成"""
    if not database.db_client or not settings.settings_manager:
        raise HTTPException(503, "DBまたは設定マネージャーが初期化されていません")

    try:
        question_text = request.question.strip()
        answer_text = request.answer.strip()
        category_name = request.category_name.strip()

        embedding = None
        try:
            embedding_model = settings.settings_manager.settings.get("embedding_model", "models/gemini-embedding-001")
            embedding_response = genai.embed_content(
                model=embedding_model,
                content=question_text
            )
            embedding = embedding_response["embedding"]
        except Exception as e:
            logging.error(f"ベクトル生成エラー: {e}")

        insert_data = {
            "question": question_text,
            "answer": answer_text,
            "category_name": category_name,
            "embedding": embedding
        }
        result = database.db_client.client.table("category_fallbacks").insert(insert_data).execute()
        return {"message": "保存しました", "fallback": result.data[0]}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"作成エラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="処理に失敗しました。")

# 修正: /api/fallbacks/{qa_id} -> /{qa_id} （リクエストは Pydantic 必須・dict 禁止）
@router.put("/{qa_id}")
async def update_fallback(qa_id: int, request: FallbackUpdate, user: dict = Depends(require_auth)):
    """Q&Aを更新（部分更新可）"""
    if not database.db_client or not settings.settings_manager:
        raise HTTPException(503, "DBまたは設定マネージャーが初期化されていません")

    update_data = request.model_dump(exclude_unset=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="更新するフィールドがありません")

    try:
        if "question" in update_data:
            new_question = (update_data["question"] or "").strip()
            if not new_question:
                raise HTTPException(status_code=400, detail="question cannot be empty")
            update_data["question"] = new_question
            embedding_model = settings.settings_manager.settings.get("embedding_model", "models/gemini-embedding-001")
            try:
                embedding_response = genai.embed_content(
                    model=embedding_model,
                    content=new_question
                )
                update_data["embedding"] = embedding_response["embedding"]
            except Exception as e:
                logging.error(f"ベクトル更新エラー: {e}")
                update_data["embedding"] = None

        result = database.db_client.client.table("category_fallbacks").update(update_data).eq("id", qa_id).execute()
        return {"message": "更新しました", "fallback": result.data[0]}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"更新エラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="処理に失敗しました。")

# 修正: /api/fallbacks/{qa_id} -> /{qa_id}
@router.delete("/{qa_id}")
async def delete_fallback(qa_id: int, user: dict = Depends(require_auth)):
    """Q&Aを削除"""
    if not database.db_client:
        raise HTTPException(503, "DB not initialized")
    try:
        database.db_client.client.table("category_fallbacks").delete().eq("id", qa_id).execute()
        return {"message": "削除しました"}
    except Exception as e:
        logging.error(f"削除エラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="処理に失敗しました。")

# 修正: /api/fallbacks/vectorize-all -> /vectorize-all
@router.post("/vectorize-all")
async def vectorize_all_missing_fallbacks(user: dict = Depends(require_auth)):
    """全Q&Aのベクトルを再生成・修復する"""
    if not database.db_client or not settings.settings_manager:
        raise HTTPException(503, "DB not initialized")

    logging.info(f"全Q&Aのベクトル修復処理を開始...")
    try:
        response = database.db_client.client.table("category_fallbacks").select("id, question").execute()
        if not response.data:
            return {"message": "データがありません。"}

        embedding_model = settings.settings_manager.settings.get("embedding_model", "models/gemini-embedding-001")
        count = 0
        
        for item in response.data:
            item_id = item['id']
            original_text = item.get('question', '')
            if not original_text: continue
            text_to_vectorize = re.sub(r'[\r\n\t]', ' ', original_text).strip()
            if not text_to_vectorize: continue

            try:
                embedding_response = genai.embed_content(
                    model=embedding_model,
                    content=text_to_vectorize
                )
                new_embedding = embedding_response["embedding"]
                database.db_client.client.table("category_fallbacks").update({
                    "embedding": new_embedding
                }).eq("id", item_id).execute()
                count += 1
                await asyncio.sleep(1)
            except Exception as e:
                if "429" in str(e): await asyncio.sleep(30)
                logging.error(f"ID {item_id} エラー: {e}")

        return {"message": f"修復完了。{count}件のベクトルを再生成しました。"}
    except Exception as e:
        logging.error(f"ベクトル修復処理エラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="処理に失敗しました。")