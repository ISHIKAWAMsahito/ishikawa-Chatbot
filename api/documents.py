import logging
import asyncio
import traceback
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Query
import google.generativeai as genai
import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel

# リクエストボディのモデル定義
class ScrapeRequest(BaseModel):
    url: str
    collection_name: str
    embedding_model: str

from core.dependencies import require_auth
from core import database
from core import settings
from core.config import ACTIVE_COLLECTION_NAME
from services import document_processor

router = APIRouter()

# 修正: /api/documents/all -> /all
@router.get("/all")
async def get_all_documents(
    user: dict = Depends(require_auth),
    page: int = Query(1, ge=1),
    limit: int = Query(100, ge=1, le=1000),
    search: Optional[str] = Query(None),
    category: Optional[str] = Query(None)
):
    """全ドキュメントをページネーション、検索、カテゴリフィルタ対応で取得"""
    if not database.db_client:
        raise HTTPException(503, "DB not initialized")
    try:
        # ベースクエリ
        query = database.db_client.client.table("documents").select("*")
        count_query = database.db_client.client.table("documents").select("id", count='exact')
        
        # フィルタ適用
        if category:
            query = query.eq("metadata->>category", category)
            count_query = count_query.eq("metadata->>category", category)

        if search:
            safe_search = search.replace('"', '""')
            search_term = f"%{safe_search}%"
            
            query = query.ilike("content", search_term)
            count_query = count_query.ilike("content", search_term)

        # カウント実行
        count_response = count_query.execute()
        total_records = count_response.count or 0

        # データ取得実行
        offset = (page - 1) * limit
        data_response = query.order("id", desc=True).range(offset, offset + limit - 1).execute()

        return {
            "documents": data_response.data or [],
            "total": total_records,
            "page": page,
            "limit": limit
        }

    except Exception as e:
        logging.error(f"ドキュメント一覧取得エラー: {e}")
        logging.error(traceback.format_exc()) 
        raise HTTPException(status_code=500, detail=str(e))

# 修正: /api/documents/{doc_id} -> /{doc_id}
@router.get("/{doc_id}")
async def get_document_by_id(doc_id: int, user: dict = Depends(require_auth)):
    """特定のドキュメントを取得"""
    if not database.db_client:
        raise HTTPException(503, "DB not initialized")
    try:
        result = database.db_client.client.table("documents").select("*").eq("id", doc_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="ドキュメントが見つかりません")
        return result.data[0]
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"ドキュメント取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 修正: /api/documents/{doc_id} -> /{doc_id}
@router.put("/{doc_id}")
async def update_document(doc_id: int, request: Dict[str, Any], user: dict = Depends(require_auth)):
    """ドキュメントを更新。content更新時にベクトルも再生成する。"""
    if not database.db_client or not settings.settings_manager:
        raise HTTPException(503, "DBまたは設定マネージャーが初期化されていません")
    try:
        update_data = {}

        if "content" in request:
            new_content = request["content"]
            update_data["content"] = new_content
            
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
        
        result = database.db_client.client.table("documents").update(update_data).eq("id", doc_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="ドキュメントが見つかりません")
        
        logging.info(f"ドキュメント {doc_id} を更新しました(管理者: {user.get('email')})")
        return {"message": "ドキュメントを更新しました", "document": result.data[0]}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"ドキュメント更新エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 修正: /api/documents/{doc_id} -> /{doc_id}
@router.delete("/{doc_id}")
async def delete_document(doc_id: int, user: dict = Depends(require_auth)):
    """ドキュメントを削除"""
    if not database.db_client:
        raise HTTPException(503, "DB not initialized")
    try:
        result = database.db_client.client.table("documents").delete().eq("id", doc_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="ドキュメントが見つかりません")
        
        logging.info(f"ドキュメント {doc_id} を削除しました(管理者: {user.get('email')})")
        return {"message": "ドキュメントを削除しました"}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"ドキュメント削除エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 修正: main.pyのprefix配下になるため変更なし (URL: /api/admin/documents/upload)
@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...), 
    category: str = Form("その他"), 
    # ▼ フロントエンドからのモデル指定を受け取る
    embedding_model: str = Form("models/gemini-embedding-001"), 
    user: dict = Depends(require_auth)
):
    """ファイルを受け取り、(古いデータを削除後)、テキスト抽出・チャンキング・ベクトル化してDBに挿入"""
    if not database.db_client or not settings.settings_manager or not document_processor.simple_processor:
        raise HTTPException(503, "システムが初期化されていません")
    
    try:
        filename = file.filename
        content = await file.read()
        
        logging.info(f"ファイルアップロード受信: {filename} (カテゴリ: {category}, モデル: {embedding_model})")

        # 1. 古いデータを削除
        logging.info(f"古いチャンク (source: {filename}) を削除しています...")
        try:
            delete_result = database.db_client.client.table("documents").delete().eq("metadata->>source", filename).execute()
            if delete_result.data:
                logging.info(f"{len(delete_result.data)} 件の古いチャンクを削除しました。")
            else:
                logging.info("削除対象の古いチャンクはありませんでした。")
        except Exception as e:
            logging.error(f"古いチャンクの削除中にエラー: {e}。処理を続行します。")
            
        # 2. 新しいデータを処理
        collection_name = settings.settings_manager.settings.get("collection", ACTIVE_COLLECTION_NAME)
        docs_to_embed = document_processor.simple_processor.process_and_chunk(filename, content, category, collection_name)
        
        if not docs_to_embed:
            raise HTTPException(status_code=400, detail="ファイルからテキストを抽出できませんでした。")

        # ---------------------------------------------------------
        # 【重要】ここに embedding_model を上書きする行が無いことを確認
        # ---------------------------------------------------------

        total_chunks = len(docs_to_embed)
        logging.info(f"{total_chunks} 件のチャンクをベクトル化・挿入します... (Model: {embedding_model})")

        # 3. バッチ処理 (API課金の節約と高速化)
        # 100件ずつまとめてAPIに送信することで、リクエスト回数を1/100に減らします
        batch_size = 100
        count = 0

        for i in range(0, total_chunks, batch_size):
            batch_docs = docs_to_embed[i : i + batch_size]
            batch_texts = [doc.page_content for doc in batch_docs]
            
            try:
                # API呼び出し (ここで100件分を 1回のリクエスト で処理します)
                # task_type="retrieval_document" は検索対象ドキュメント用として精度が向上します
                embedding_response = genai.embed_content(
                    model=embedding_model,
                    content=batch_texts,
                    task_type="retrieval_document"
                )
                embeddings = embedding_response["embedding"]
                
                # 取得したベクトルをDBに保存
                for j, doc in enumerate(batch_docs):
                    database.db_client.insert_document(
                        content=doc.page_content, 
                        embedding=embeddings[j], 
                        metadata=doc.metadata
                    )
                    count += 1
                
                logging.info(f"バッチ処理進行中: {count}/{total_chunks}")
                # API制限対策の待機 (バッチ処理なら0.5秒でも十分安全です)
                await asyncio.sleep(0.5)

            except Exception as e:
                # クォータエラーなどが起きた場合の待機処理
                if "429" in str(e) or "quota" in str(e).lower():
                    logging.warning(f"API制限を検知しました。10秒待機してリトライします... ({i}~)")
                    await asyncio.sleep(10)
                    try:
                        # 1回だけリトライ
                        embedding_response = genai.embed_content(
                            model=embedding_model,
                            content=batch_texts,
                            task_type="retrieval_document"
                        )
                        embeddings = embedding_response["embedding"]
                        for j, doc in enumerate(batch_docs):
                            database.db_client.insert_document(
                                content=doc.page_content, 
                                embedding=embeddings[j], 
                                metadata=doc.metadata
                            )
                            count += 1
                    except Exception as retry_e:
                        logging.error(f"リトライも失敗しました: {retry_e}")
                        continue
                else:
                    logging.error(f"バッチ処理エラー (index {i}~): {e}")
                    await asyncio.sleep(2)
                    continue

        logging.info(f"ファイル処理完了: {filename} ({count}/{total_chunks}件のチャンクをDBに挿入)")
        return {"chunks": count, "filename": filename, "total": total_chunks}

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"ファイルアップロード処理エラー: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# 修正: main.pyのprefix配下になる (URL: /api/admin/documents/collections/...)
@router.get("/collections/{collection_name}/documents")
async def get_documents(collection_name: str):
    if not database.db_client:
        raise HTTPException(503, "DB not initialized")
    return {
        "documents": database.db_client.get_documents_by_collection(collection_name),
        "count": database.db_client.count_chunks_in_collection(collection_name)
    }

# 修正: main.pyのprefix配下になる (URL: /api/admin/documents/scrape)
@router.post("/scrape")
async def scrape_website(
    request: ScrapeRequest, 
    user: dict = Depends(require_auth)
):
    """URLからテキストを抽出し、(古いデータを削除後)、ベクトル化してDBに挿入"""
    if not database.db_client or not settings.settings_manager or not document_processor.simple_processor:
        raise HTTPException(503, "システムが初期化されていません")

    logging.info(f"Scrapeリクエスト受信: {request.url} (Collection: {request.collection_name})")

    try:
        # 1. Webサイトからコンテンツを取得
        async with httpx.AsyncClient(verify=False) as client:
            try:
                response = await client.get(request.url, follow_redirects=True, timeout=10.0)
                response.raise_for_status() # 200 OK以外はエラー
            except httpx.RequestError as e:
                logging.error(f"URL取得エラー: {e}")
                raise HTTPException(status_code=400, detail=f"URLの取得に失敗しました: {e}")
        
        # 2. HTMLからテキストを抽出
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        
        target_element = soup.body
        if not target_element:
            logging.warning(f"URL: {request.url} に <body> タグが見つからなかったため、HTML全体からテキストを抽出します。")
            target_element = soup 

        body_text = target_element.get_text(separator=' ', strip=True) 

        if not body_text:
            raise HTTPException(status_code=400, detail="Webサイトからテキストを抽出できませんでした。")
        
        # 3. チャンキング 
        filename_from_url = request.url.split('/')[-1] or request.url
        source_name = f"scrape_{filename_from_url}.txt"

        # 3b. 古いデータを削除
        logging.info(f"古いチャンク (source: {source_name}) を削除しています...")
        try:
            delete_result = database.db_client.client.table("documents").delete().eq("metadata->>source", source_name).execute()
            if delete_result.data:
                logging.info(f"{len(delete_result.data)} 件の古いチャンクを削除しました。")
            else:
                logging.info("削除対象の古いチャンクはありませんでした。")
        except Exception as e:
            logging.error(f"古いチャンクの削除中にエラー: {e}。処理を続行します。")

        # 3c. 新しいデータを処理
        docs_to_embed = document_processor.simple_processor.process_and_chunk(
            filename=source_name, 
            content=body_text.encode('utf-8'), 
            category="WebScrape", 
            collection_name=request.collection_name
        )
        
        if not docs_to_embed:
            raise HTTPException(status_code=400, detail="テキストのチャンキングに失敗しました。")

        # 4. ベクトル化 & DB挿入
        embedding_model = request.embedding_model
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
                
                database.db_client.insert_document(
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
                    database.db_client.insert_document(doc.page_content, embedding, doc.metadata)
                else:
                    logging.error(f"チャンク処理エラー ({request.url}): {e}")
                    continue

        logging.info(f"スクレイプ処理完了: {request.url} ({count}/{total_chunks}件のチャンクをDBに挿入)")
        return {"chunks": count, "filename": request.url, "total": total_chunks}

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"スクレイプ処理エラー: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))