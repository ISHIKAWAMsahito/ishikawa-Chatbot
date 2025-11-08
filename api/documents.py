import logging
import asyncio
import traceback
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Query
import google.generativeai as genai

from core.dependencies import require_auth
from core.database import db_client
from core.settings import settings_manager
from core.config import ACTIVE_COLLECTION_NAME
from services.document_processor import simple_processor

router = APIRouter()

@router.get("/api/documents/all")
async def get_all_documents(
    user: dict = Depends(require_auth),
    page: int = Query(1, ge=1),
    limit: int = Query(100, ge=1, le=1000),
    search: Optional[str] = Query(None),
    category: Optional[str] = Query(None)
):
    """全ドキュメントをページネーション、検索、カテゴリフィルタ対応で取得"""
    if not db_client:
        raise HTTPException(503, "DB not initialized")
    try:
        query = db_client.client.table("documents")
        count_query = db_client.client.table("documents")
        
        if category:
            query = query.eq("metadata->>category", category)
            count_query = count_query.eq("metadata->>category", category)

        if search:
            safe_search = search.replace('"', '""')
            search_term = f"*{safe_search}*"
            or_filter_string = f"content.ilike.{search_term}"
            
            query = query.or_(or_filter_string)
            count_query = count_query.or_(or_filter_string)

        count_response = count_query.select("id", count='exact').execute()
        total_records = count_response.count or 0

        offset = (page - 1) * limit
        data_response = query.select("*").order("id", desc=True).range(offset, offset + limit - 1).execute()

        return {
            "documents": data_response.data or [],
            "total": total_records,
            "page": page,
            "limit": limit
        }

    except Exception as e:
        logging.error(f"ドキュメント一覧取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/documents/{doc_id}")
async def get_document_by_id(doc_id: int, user: dict = Depends(require_auth)):
    """特定のドキュメントを取得"""
    if not db_client:
        raise HTTPException(503, "DB not initialized")
    try:
        result = db_client.client.table("documents").select("*").eq("id", doc_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="ドキュメントが見つかりません")
        return result.data[0]
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"ドキュメント取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/api/documents/{doc_id}")
async def update_document(doc_id: int, request: Dict[str, Any], user: dict = Depends(require_auth)):
    """ドキュメントを更新。content更新時にベクトルも再生成する。"""
    if not db_client or not settings_manager:
        raise HTTPException(503, "DBまたは設定マネージャーが初期化されていません")
    try:
        update_data = {}

        if "content" in request:
            new_content = request["content"]
            update_data["content"] = new_content
            
            embedding_model = settings_manager.settings.get("embedding_model", "text-embedding-004")
            
            logging.info(f"ドキュメント {doc_id} のコンテンツが変更されたため、ベクトルを再生成します...")
            try:
                embedding_response = genai.embed_content(
                    model=embedding_model,
                    content=new_content
                )
                update_data["embedding"] = embedding_response["embedding"]
                logging.info(f"ドキュメント {doc_id} のベクトル再生成が完了しました。")
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    logging.warning("ベクトル再生成でAPI制限に達しました。30秒待機します。")
                    await asyncio.sleep(30)
                    embedding_response = genai.embed_content(
                        model=embedding_model,
                        content=new_content
                    )
                    update_data["embedding"] = embedding_response["embedding"]
                else:
                    logging.error(f"ベクトル再生成エラー: {e}")
                    raise HTTPException(status_code=500, detail=f"ベクトル再生成中にエラーが発生しました: {e}")

        if "metadata" in request:
            update_data["metadata"] = request["metadata"]

        if not update_data:
            raise HTTPException(status_code=400, detail="更新するデータがありません")
        
        result = db_client.client.table("documents").update(update_data).eq("id", doc_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="ドキュメントが見つかりません")
        
        logging.info(f"ドキュメント {doc_id} を更新しました(管理者: {user.get('email')})")
        return {"message": "ドキュメントを更新しました", "document": result.data[0]}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"ドキュメント更新エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: int, user: dict = Depends(require_auth)):
    """ドキュメントを削除"""
    if not db_client:
        raise HTTPException(503, "DB not initialized")
    try:
        result = db_client.client.table("documents").delete().eq("id", doc_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="ドキュメントが見つかりません")
        
        logging.info(f"ドキュメント {doc_id} を削除しました(管理者: {user.get('email')})")
        return {"message": "ドキュメントを削除しました"}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"ドキュメント削除エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...), 
    category: str = Form("その他"), 
    user: dict = Depends(require_auth)
):
    """ファイルを受け取り、テキスト抽出・チャンキング・ベクトル化してDBに挿入"""
    if not db_client or not settings_manager or not simple_processor:
        raise HTTPException(503, "システムが初期化されていません")

    try:
        filename = file.filename
        content = await file.read()
        
        logging.info(f"ファイルアップロード受信: {filename} (カテゴリ: {category})")

        collection_name = settings_manager.settings.get("collection", ACTIVE_COLLECTION_NAME)
        docs_to_embed = simple_processor.process_and_chunk(filename, content, category, collection_name)
        
        if not docs_to_embed:
            raise HTTPException(status_code=400, detail="ファイルからテキストを抽出できませんでした。")

        embedding_model = settings_manager.settings.get("embedding_model", "text-embedding-004")
        total_chunks = len(docs_to_embed)
        logging.info(f"{total_chunks} 件のチャンクをベクトル化・挿入します...")

        count = 0
        for doc in docs_to_embed:
            try:
                embedding_response = genai.embed_content(
                    model=embedding_model, 
                    content=doc.page_content
                )
                embedding = embedding_response["embedding"]
                
                db_client.insert_document(
                    content=doc.page_content, 
                    embedding=embedding, 
                    metadata=doc.metadata
                )
                count += 1
                
                await asyncio.sleep(1)
            
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    logging.warning("埋め込み生成でAPI制限に達しました。30秒待機します。")
                    await asyncio.sleep(30)
                    embedding_response = genai.embed_content(model=embedding_model, content=doc.page_content)
                    embedding = embedding_response["embedding"]
                    db_client.insert_document(doc.page_content, embedding, doc.metadata)
                else:
                    logging.error(f"チャンク処理エラー ({filename}): {e}")
                    continue

        logging.info(f"ファイル処理完了: {filename} ({count}/{total_chunks}件のチャンクをDBに挿入)")
        return {"chunks": count, "filename": filename, "total": total_chunks}

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"ファイルアップロード処理エラー: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/collections/{collection_name}/documents")
async def get_documents(collection_name: str):
    if not db_client:
        raise HTTPException(503, "DB not initialized")
    return {
        "documents": db_client.get_documents_by_collection(collection_name),
        "count": db_client.count_chunks_in_collection(collection_name)
    }