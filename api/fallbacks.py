import logging
import asyncio
import re # 正規表現用
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
import google.generativeai as genai
logging.info(f"全Q&Aのベクトル修復処理を開始...")
from core.dependencies import require_auth
from core import database
from core import settings
from core.config import GEMINI_API_KEY

router = APIRouter()

# APIキーの設定
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

@router.get("/api/fallbacks")
async def get_all_fallbacks(user: dict = Depends(require_auth)):
    """Q&A(フォールバック)をすべて取得"""
    if not database.db_client:
        raise HTTPException(503, "DB not initialized")
    try:
        result = database.db_client.client.table("category_fallbacks").select("id, question, answer, category_name, embedding").order("id", desc=True).execute()
        return {"fallbacks": result.data or []}
    except Exception as e:
        logging.error(f"Q&A一覧取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/fallbacks")
async def create_fallback(request: Dict[str, Any], user: dict = Depends(require_auth)):
    """新しいQ&Aを作成"""
    if not database.db_client or not settings.settings_manager: 
        raise HTTPException(503, "DBまたは設定マネージャーが初期化されていません")
    
    try:
        question_text = request.get("question")
        answer_text = request.get("answer")
        category_name = request.get("category_name")
        
        if not question_text or not answer_text or not category_name:
             raise HTTPException(status_code=400, detail="必須項目が不足しています")

        embedding = None
        try:
            embedding_model = settings.settings_manager.settings.get("embedding_model", "text-embedding-004")
            
            # テキストのクリーニング
            clean_question = question_text.strip()
            
            embedding_response = genai.embed_content(
                model=embedding_model,
                content=clean_question
            )
            embedding = embedding_response["embedding"]
        
        except Exception as e:
            logging.error(f"ベクトル生成エラー: {e}")
            # 失敗しても保存はする

        insert_data = {
            "question": question_text,
            "answer": answer_text,
            "category_name": category_name,
            "embedding": embedding
        }
        
        result = database.db_client.client.table("category_fallbacks").insert(insert_data).execute()
        return {"message": "保存しました", "fallback": result.data[0]}
    
    except Exception as e:
        logging.error(f"作成エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/api/fallbacks/{qa_id}")
async def update_fallback(qa_id: int, request: Dict[str, Any], user: dict = Depends(require_auth)):
    """Q&Aを更新"""
    if not database.db_client or not settings.settings_manager:
        raise HTTPException(503, "DBまたは設定マネージャーが初期化されていません")
    try:
        update_data = {}
        
        if "question" in request:
            new_question = request["question"]
            if not new_question or not new_question.strip():
                raise HTTPException(status_code=400, detail="question cannot be empty")
            
            update_data["question"] = new_question
            
            embedding_model = settings.settings_manager.settings.get("embedding_model", "text-embedding-004")
            try:
                clean_question = new_question.strip()
                embedding_response = genai.embed_content(
                    model=embedding_model,
                    content=clean_question
                )
                update_data["embedding"] = embedding_response["embedding"]
            except Exception as e:
                logging.error(f"ベクトル更新エラー: {e}")
                update_data["embedding"] = None

        if "answer" in request:
            update_data["answer"] = request["answer"]
        if "category_name" in request:
            update_data["category_name"] = request["category_name"]

        result = database.db_client.client.table("category_fallbacks").update(update_data).eq("id", qa_id).execute()
        return {"message": "更新しました", "fallback": result.data[0]}
    except Exception as e:
        logging.error(f"更新エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/api/fallbacks/{qa_id}")
async def delete_fallback(qa_id: int, user: dict = Depends(require_auth)):
    """Q&Aを削除"""
    if not database.db_client:
        raise HTTPException(503, "DB not initialized")
    try:
        database.db_client.client.table("category_fallbacks").delete().eq("id", qa_id).execute()
        return {"message": "削除しました"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ▼▼▼▼ 【ここを正常版に戻しました】 ▼▼▼▼
@router.post("/api/fallbacks/vectorize-all")
async def vectorize_all_missing_fallbacks(user: dict = Depends(require_auth)):
    """全Q&Aのベクトルを再生成・修復する"""
    if not database.db_client or not settings.settings_manager:
        raise HTTPException(503, "DB not initialized")

    logging.info(f"全Q&Aのベクトル修復処理を開始...")
    
    try:
        # 全データを取得
        response = database.db_client.client.table("category_fallbacks").select("id, question").execute()
        
        if not response.data:
            return {"message": "データがありません。"}

        embedding_model = settings.settings_manager.settings.get("embedding_model", "text-embedding-004")
        count = 0
        
        for item in response.data:
            item_id = item['id']
            original_text = item.get('question', '')

            # ★重要: テキストのクリーニング（空文字チェックとゴミ取り）
            if not original_text:
                continue
                
            # 改行やタブをスペースに置換し、前後の空白を削除
            text_to_vectorize = re.sub(r'[\r\n\t]', ' ', original_text).strip()

            if not text_to_vectorize:
                continue

            try:
                # ★修正: ちゃんと「質問文」をAPIに送る（テスト文字列ではない）
                embedding_response = genai.embed_content(
                    model=embedding_model,
                    content=text_to_vectorize
                )
                new_embedding = embedding_response["embedding"]

                database.db_client.client.table("category_fallbacks").update({
                    "embedding": new_embedding
                }).eq("id", item_id).execute()
                
                logging.info(f"ID {item_id}: ベクトル化完了 (text: {text_to_vectorize[:10]}...)")
                count += 1
                await asyncio.sleep(1) # レート制限対策

            except Exception as e:
                if "429" in str(e):
                    await asyncio.sleep(30)
                logging.error(f"ID {item_id} エラー: {e}")

        return {"message": f"修復完了。{count}件のベクトルを正しい質問文で再生成しました。"}

    except Exception as e:
        logging.error(f"処理エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))