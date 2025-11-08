import logging
import asyncio
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
import google.generativeai as genai

from core.dependencies import require_auth
from core.database import db_client
from core.settings import settings_manager

router = APIRouter()

@router.get("/api/fallbacks")
async def get_all_fallbacks(user: dict = Depends(require_auth)):
    """Q&A(フォールバック)をすべて取得"""
    if not db_client:
        raise HTTPException(503, "DB not initialized")
    try:
        result = db_client.client.table("category_fallbacks").select("*").order("id", desc=True).execute()
        return {"fallbacks": result.data or []}
    except Exception as e:
        logging.error(f"Q&A一覧取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/fallbacks")
async def create_fallback(request: Dict[str, Any], user: dict = Depends(require_auth)):
    """新しいQ&Aを作成(保存時に自動でベクトル化)"""
    if not db_client or not settings_manager: 
        raise HTTPException(503, "DBまたは設定マネージャーが初期化されていません")
    
    try:
        new_qa_text = request.get("static_response", "")
        category_name = request.get("category_name") 

        if not new_qa_text:
            raise HTTPException(status_code=400, detail="static_response (Q&Aテキスト) は必須です")
        
        if not category_name:
            raise HTTPException(status_code=400, detail="category_name は必須です")

        embedding = None
        try:
            embedding_model = settings_manager.settings.get("embedding_model", "text-embedding-004")
            logging.info(f"新規Q&Aのベクトルを生成します...")
            
            embedding_response = genai.embed_content(
                model=embedding_model,
                content=new_qa_text
            )
            embedding = embedding_response["embedding"]
            logging.info(f"新規Q&Aのベクトル生成が完了しました。")
        
        except Exception as e:
            logging.error(f"新規Q&Aのベクトル生成エラー: {e}")
            logging.warning(f"ベクトル化に失敗しましたが、テキストは保存します。")

        insert_data = {
            "static_response": new_qa_text,
            "category_name": category_name,
            "url_to_summarize": request.get("url_to_summarize"),
            "embedding": embedding
        }
        
        result = db_client.client.table("category_fallbacks").insert(insert_data).execute()
        
        if not result.data:
            raise HTTPException(status_code=500, detail="Q&Aの作成に失敗しました")

        logging.info(f"新規Q&A {result.data[0]['id']} を作成しました(管理者: {user.get('email')})")
        
        message = "新しいQ&Aを保存し、ベクトル化も完了しました。" if embedding else "新しいQ&Aを保存しました(ベクトル化には失敗)"
        return {"message": message, "fallback": result.data[0]}
    
    except HTTPException:
        raise
    except Exception as e:
        if "23502" in str(e) and "category_name" in str(e):
             raise HTTPException(status_code=400, detail="category_name は必須です (DB Error 23502)")
        logging.error(f"Q&A作成エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/api/fallbacks/{qa_id}")
async def update_fallback(qa_id: int, request: Dict[str, Any], user: dict = Depends(require_auth)):
    """Q&Aを更新(テキスト変更時に自動でベクトル化)"""
    if not db_client or not settings_manager:
        raise HTTPException(503, "DBまたは設定マネージャーが初期化されていません")
    try:
        update_data = {}
        
        if "static_response" in request:
            new_content = request["static_response"]
            update_data["static_response"] = new_content
            
            embedding_model = settings_manager.settings.get("embedding_model", "text-embedding-004")
            logging.info(f"Q&A {qa_id} のテキストが変更されたため、ベクトルを再生成します...")
            try:
                embedding_response = genai.embed_content(
                    model=embedding_model,
                    content=new_content
                )
                update_data["embedding"] = embedding_response["embedding"]
                logging.info(f"Q&A {qa_id} のベクトル再生成が完了しました。")
            except Exception as e:
                logging.error(f"Q&Aベクトル再生成エラー: {e}")
                update_data["embedding"] = None
                logging.warning(f"Q&A {qa_id} のベクトル化に失敗しましたが、テキストは更新します。")

        if "url_to_summarize" in request:
            update_data["url_to_summarize"] = request.get("url_to_summarize")
            
        if "category_name" in request:
            new_category = request.get("category_name")
            if not new_category or not new_category.strip():
                 raise HTTPException(status_code=400, detail="category_name を空にすることはできません")
            update_data["category_name"] = new_category

        if not update_data:
            raise HTTPException(status_code=400, detail="更新するデータがありません")
        
        result = db_client.client.table("category_fallbacks").update(update_data).eq("id", qa_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Q&Aが見つかりません")
        
        logging.info(f"Q&A {qa_id} を更新しました(管理者: {user.get('email')})")
        return {"message": "Q&Aを更新しました", "fallback": result.data[0]}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Q&A更新エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/api/fallbacks/{qa_id}")
async def delete_fallback(qa_id: int, user: dict = Depends(require_auth)):
    """Q&Aを削除"""
    if not db_client:
        raise HTTPException(503, "DB not initialized")
    try:
        result = db_client.client.table("category_fallbacks").delete().eq("id", qa_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Q&Aが見つかりません")
        
        logging.info(f"Q&A {qa_id} を削除しました(管理者: {user.get('email')})")
        return {"message": "Q&Aを削除しました"}
    except Exception as e:
        logging.error(f"Q&A削除エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/fallbacks/vectorize-all")
async def vectorize_all_missing_fallbacks(user: dict = Depends(require_auth)):
    """embedding が NULL のQ&Aをすべてベクトル化する"""
    if not db_client or not settings_manager:
        raise HTTPException(503, "DBまたは設定マネージャーが初期化されていません")

    logging.info(f"全Q&Aのベクトル化処理を開始...(管理者: {user.get('email')})")
    
    try:
        response = db_client.client.table("category_fallbacks").select("id, static_response").is_("embedding", "null").execute()
        
        if not response.data:
            return {"message": "ベクトル化が必要なQ&Aはありませんでした。"}

        embedding_model = settings_manager.settings.get("embedding_model", "text-embedding-004")
        count = 0
        
        for item in response.data:
            item_id = item['id']
            text_to_vectorize = item['static_response']

            if not text_to_vectorize or not text_to_vectorize.strip():
                logging.warning(f"Q&A ID {item_id}: テキストが空のためスキップします。")
                continue

            try:
                embedding_response = genai.embed_content(
                    model=embedding_model,
                    content=text_to_vectorize
                )
                new_embedding = embedding_response["embedding"]

                db_client.client.table("category_fallbacks").update({
                    "embedding": new_embedding
                }).eq("id", item_id).execute()
                
                logging.info(f"Q&A ID {item_id}: ベクトル化完了。")
                count += 1
                await asyncio.sleep(1)

            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    logging.warning(f"APIレート制限のため30秒待機します... (ID {item_id})")
                    await asyncio.sleep(30)
                    embedding_response = genai.embed_content(model=embedding_model, content=text_to_vectorize)
                    new_embedding = embedding_response["embedding"]
                    db_client.client.table("category_fallbacks").update({"embedding": new_embedding}).eq("id", item_id).execute()
                    logging.info(f"Q&A ID {item_id}: (再試行) ベクトル化完了。")
                    count += 1
                else:
                    logging.error(f"Q&A ID {item_id} のベクトル化エラー: {e}")

        logging.info(f"全Q&Aベクトル化処理完了。 {count}件を処理しました。")
        return {"message": f"ベクトル化処理が完了しました。{count}件のQ&Aを更新しました。"}

    except Exception as e:
        logging.error(f"全Q&Aベクトル化処理中にエラーが発生: {e}")
        raise HTTPException(status_code=500, detail=str(e))