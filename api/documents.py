import logging
import asyncio
import traceback
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Query
import google.generativeai as genai
import httpx  # Webアクセス用
from bs4 import BeautifulSoup # HTML解析用
from pydantic import BaseModel # リクエストの型定義用

# [documents.py の import 文のすぐ下あたり]
class ScrapeRequest(BaseModel):
    url: str
    collection_name: str
    embedding_model: str

from core.dependencies import require_auth
# ↓↓↓ [修正] モジュール本体をインポート
from core import database
from core import settings
# ↑↑↑ [修正]
from core.config import ACTIVE_COLLECTION_NAME
# ↓↓↓ [修正] モジュール本体をインポート
from services import document_processor
# ↑↑↑ [修正]

router = APIRouter()

# [documents.py]

@router.get("/api/documents/all")
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
        # 1. [修正] ベースクエリは .select() から開始する
        query = database.db_client.client.table("documents").select("*")
        
        # 2. [修正] カウント用クエリも .select() から開始する
        #    (count='exact' は select メソッド内で指定)
        count_query = database.db_client.client.table("documents").select("id", count='exact')
        
        # 3. [順序変更] .select() の *後* で .eq() を適用
        if category:
            query = query.eq("metadata->>category", category)
            count_query = count_query.eq("metadata->>category", category)

        # 4. [順序変更 & 修正] .select() の *後* で .ilike() を適用
        if search:
            safe_search = search.replace('"', '""')
            # [修正] SQLのワイルドカードは * ではなく %
            search_term = f"%{safe_search}%"
            
            # [これが正しい .ilike() の使い方]
            query = query.ilike("content", search_term)
            count_query = count_query.ilike("content", search_term)

        # 5. [修正] 既に .select() 済みなので、ここでは .execute() のみ
        count_response = count_query.execute()
        total_records = count_response.count or 0

        # 6. [修正] 既に .select() 済みなので、ここではチェーンを続ける
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
        # エラーの詳細をログに出力
        logging.error(traceback.format_exc()) 
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/documents/{doc_id}")
async def get_document_by_id(doc_id: int, user: dict = Depends(require_auth)):
    """特定のドキュメントを取得"""
    # ↓↓↓ [修正] "database." を追加
    if not database.db_client:
        raise HTTPException(503, "DB not initialized")
    try:
        result = database.db_client.client.table("documents").select("*").eq("id", doc_id).execute()
        # ↑↑↑ [修正] "database." を追加
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
    # ↓↓↓ [修正] "database." と "settings." を追加
    if not database.db_client or not settings.settings_manager:
        raise HTTPException(503, "DBまたは設定マネージャーが初期化されていません")
    try:
        update_data = {}

        if "content" in request:
            new_content = request["content"]
            update_data["content"] = new_content
            
            embedding_model = settings.settings_manager.settings.get("embedding_model", "text-embedding-004")
            # ↑↑↑ [修正] "settings." を追加
            
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
        
        # ↓↓↓ [修正] "database." を追加
        result = database.db_client.client.table("documents").update(update_data).eq("id", doc_id).execute()
        # ↑↑↑ [修正] "database." を追加
        
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
    # ↓↓↓ [修正] "database." を追加
    if not database.db_client:
        raise HTTPException(503, "DB not initialized")
    try:
        result = database.db_client.client.table("documents").delete().eq("id", doc_id).execute()
        # ↑↑↑ [修正] "database." を追加
        
        if not result.data:
            raise HTTPException(status_code=404, detail="ドキュメントが見つかりません")
        
        logging.info(f"ドキュメント {doc_id} を削除しました(管理者: {user.get('email')})")
        return {"message": "ドキュメントを削除しました"}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"ドキュメント削除エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
        # ... (1. Webサイトからコンテンツを取得) ...
        # ... (2. HTMLからテキストを抽出) ...
        # ... (body_text = ... の行まで) ...

        if not body_text:
            raise HTTPException(status_code=400, detail="Webサイトからテキストを抽出できませんでした。")
        
        # 3. チャンキング (uploadのロジックを流用)
        filename_from_url = request.url.split('/')[-1] or request.url
        
        # ▼▼▼ [修正] source_nameを変数として定義 ▼▼▼
        source_name = f"scrape_{filename_from_url}.txt"

        # ▼▼▼ [ここから追加] ▼▼▼
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
        # ▲▲▲ [ここまで追加] ▲▲▲

        # 3c. 新しいデータを処理
        docs_to_embed = document_processor.simple_processor.process_and_chunk(
            filename=source_name, # ⬅️ [修正] 変数を使う
            content=body_text.encode('utf-8'), # process_and_chunk は bytes を期待
            category="WebScrape", 
            collection_name=request.collection_name
        )
        
        if not docs_to_embed:
            raise HTTPException(status_code=400, detail="ファイルからテキストを抽出できませんでした。")

        embedding_model = settings.settings_manager.settings.get("embedding_model", "text-embedding-004")
        # ↑↑↑ [修正] "settings." を追加
        
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
                
                # ↓↓↓ [修正] "database." を追加
                database.db_client.insert_document(
                    content=doc.page_content, 
                    embedding=embedding, 
                    metadata=doc.metadata
                )
                # ↑↑↑ [修正]
                count += 1
                
                await asyncio.sleep(1)
            
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    logging.warning("埋め込み生成でAPI制限に達しました。30秒待機します。")
                    await asyncio.sleep(30)
                    embedding_response = genai.embed_content(model=embedding_model, content=doc.page_content)
                    embedding = embedding_response["embedding"]
                    # ↓↓↓ [修正] "database." を追加
                    database.db_client.insert_document(doc.page_content, embedding, doc.metadata)
                    # ↑↑↑ [修正]
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
    # ↓↓↓ [修正] "database." を追加
    if not database.db_client:
        raise HTTPException(503, "DB not initialized")
    return {
        "documents": database.db_client.get_documents_by_collection(collection_name),
        "count": database.db_client.count_chunks_in_collection(collection_name)
    }

@router.post("/scrape")
async def scrape_website(
    request: ScrapeRequest, 
    user: dict = Depends(require_auth)
):
    """URLからテキストを抽出し、ベクトル化してDBに挿入"""
    # ↓↓↓ [修正] "document_processor." を追加
    if not database.db_client or not settings.settings_manager or not document_processor.simple_processor:
        raise HTTPException(503, "システムが初期化されていません")
    # ↑↑↑ [修正]

    logging.info(f"Scrapeリクエスト受信: {request.url} (Collection: {request.collection_name})")

    try:
        # 1. Webサイトからコンテンツを取得
        async with httpx.AsyncClient() as client:
            try:
                # タイムアウトとリダイレクトを許可
                response = await client.get(request.url, follow_redirects=True, timeout=10.0)
                response.raise_for_status() # 200 OK以外はエラー
            except httpx.RequestError as e:
                logging.error(f"URL取得エラー: {e}")
                raise HTTPException(status_code=400, detail=f"URLの取得に失敗しました: {e}")
        
        # 2. HTMLからテキストを抽出 (堅牢性を向上)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # scriptやstyleタグは除く
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        
        # soup.body が存在するかチェックする
        target_element = soup.body
        if not target_element:
            # body がない場合 (例: <frameset> のみ)、soup全体をフォールバックとして使用
            logging.warning(f"URL: {request.url} に <body> タグが見つからなかったため、HTML全体からテキストを抽出します。")
            target_element = soup 

        body_text = target_element.get_text(separator=' ', strip=True)

        if not body_text:
            raise HTTPException(status_code=400, detail="Webサイトからテキストを抽出できませんでした。")
        
        # 3. チャンキング (uploadのロジックを流用)
        # ファイル名の代わりにURLを、カテゴリは"WebScrape"にする
        filename_from_url = request.url.split('/')[-1] or request.url
        # ↓↓↓ [修正] "document_processor." を追加
        docs_to_embed = document_processor.simple_processor.process_and_chunk(
            filename=f"scrape_{filename_from_url}.txt", 
            content=body_text.encode('utf-8'), # process_and_chunk は bytes を期待
            category="WebScrape", 
            collection_name=request.collection_name
        )
        # ↑↑↑ [修正]
        
        if not docs_to_embed:
            raise HTTPException(status_code=400, detail="テキストのチャンキングに失敗しました。")

        # 4. ベクトル化 & DB挿入 (uploadのロジックを流用)
        embedding_model = request.embedding_model # リクエストで指定されたモデルを使用
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
                
                # [修正] "database." を追加
                database.db_client.insert_document(
                    content=doc.page_content, 
                    embedding=embedding, 
                    metadata=doc.metadata
                )
                count += 1
                
                # レート制限を避けるための軽い待機
                await asyncio.sleep(1) # /upload と同様
            
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    logging.warning("埋め込み生成でAPI制限に達しました。30秒待機します。")
                    await asyncio.sleep(30)
                    embedding_response = genai.embed_content(model=embedding_model, content=doc.page_content)
                    embedding = embedding_response["embedding"]
                    database.db_client.insert_document(doc.page_content, embedding, doc.metadata) # [修正] "database." を追加
                else:
                    logging.error(f"チャンク処理エラー ({request.url}): {e}")
                    continue

        logging.info(f"スクレイプ処理完了: {request.url} ({count}/{total_chunks}件のチャンクをDBに挿入)")
        # admin.html が期待するレスポンス
        return {"chunks": count, "filename": request.url, "total": total_chunks}

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"スクレイプ処理エラー: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))