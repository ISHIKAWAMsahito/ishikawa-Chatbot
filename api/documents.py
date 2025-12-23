import logging
import asyncio
import traceback
from typing import Dict, Any, Optional, List
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

# RAGの検索対象となるドキュメント（知識データ）を管理します。

# データソース:

# ファイルアップロード: PDFやテキストファイルを読み込みます。

# Webスクレイピング: 指定されたURLからテキストを抽出します。

# メモリ対策: batch_size = 50 などで小分けに処理しており、Renderなどのメモリ制限が厳しい環境でも落ちないよう工夫されています。

# ベクトル化: コンテンツが追加・更新されると、自動的に genai.embed_content を呼んでベクトルデータを生成・DB保存しています。

# ----------------------------------------------------------------
# 読み取り・更新・削除系
# ----------------------------------------------------------------

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


@router.put("/{doc_id}")
async def update_document(doc_id: int, request: Dict[str, Any], user: dict = Depends(require_auth)):
    """ドキュメントを更新。content更新時にベクトルも再生成する。"""
    if not database.db_client or not settings.settings_manager:
        raise HTTPException(503, "DBまたは設定マネージャーが初期化されていません")
    try:
        # 現在の設定から埋め込みモデルを取得（リクエストに含まれていなければデフォルト使用）
        embedding_model = request.get("embedding_model", "models/gemini-embedding-001")
        
        update_data = {}

        if "content" in request:
            new_content = request["content"]
            update_data["content"] = new_content
            
            logging.info(f"ドキュメント {doc_id} のコンテンツが変更されたため、ベクトルを再生成します...")
            try:
                embedding_response = genai.embed_content(
                    model=embedding_model,
                    content=new_content,
                    task_type="retrieval_document" 
                )
                update_data["embedding"] = embedding_response["embedding"]
                logging.info(f"ドキュメント {doc_id} のベクトル再生成が完了しました。")
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    logging.warning("ベクトル再生成でAPI制限に達しました。30秒待機します。")
                    await asyncio.sleep(30)
                    embedding_response = genai.embed_content(
                        model=embedding_model,
                        content=new_content,
                        task_type="retrieval_document"
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


# ----------------------------------------------------------------
# アップロード & スクレイピング (Renderメモリ対策版)
# ----------------------------------------------------------------

async def process_batch_insert(batch_docs: List[Any], embedding_model: str, collection_name: str):
    """
    バッチ処理用のヘルパー関数。
    受け取ったドキュメントのリストをベクトル化し、DBに挿入する。
    """
    if not batch_docs:
        return 0

    batch_texts = [doc.page_content for doc in batch_docs]
    inserted_count = 0

    try:
        # API呼び出し (まとめて処理)
        embedding_response = genai.embed_content(
            model=embedding_model,
            content=batch_texts,
            task_type="retrieval_document"
        )
        embeddings = embedding_response["embedding"]
        
        # DBへの挿入
        for j, doc in enumerate(batch_docs):
            if "collection_name" not in doc.metadata:
                doc.metadata["collection_name"] = collection_name
            
            database.db_client.insert_document(
                content=doc.page_content, 
                embedding=embeddings[j], 
                metadata=doc.metadata
            )
            inserted_count += 1
            
        return inserted_count

    except Exception as e:
        # クォータエラー時のリトライ処理
        if "429" in str(e) or "quota" in str(e).lower():
            logging.warning("API制限(429/Quota)を検知。15秒待機してリトライします...")
            await asyncio.sleep(15)
            try:
                # リトライ
                embedding_response = genai.embed_content(
                    model=embedding_model,
                    content=batch_texts,
                    task_type="retrieval_document"
                )
                embeddings = embedding_response["embedding"]
                for j, doc in enumerate(batch_docs):
                    if "collection_name" not in doc.metadata:
                        doc.metadata["collection_name"] = collection_name
                    
                    database.db_client.insert_document(
                        content=doc.page_content, 
                        embedding=embeddings[j], 
                        metadata=doc.metadata
                    )
                    inserted_count += 1
                return inserted_count
            except Exception as retry_e:
                logging.error(f"リトライも失敗しました: {retry_e}")
                raise retry_e
        else:
            logging.error(f"バッチベクトル化エラー: {e}")
            raise e


@router.post("/upload")#12/24 GeminiAPIをたたいてスクレイピングできるように改良
async def upload_document(
    file: UploadFile = File(...), 
    category: str = Form("その他"), 
    embedding_model: str = Form("models/gemini-embedding-001"), 
    user: dict = Depends(require_auth)
):
    """ファイルを受け取り、メモリ効率よくバッチ処理でチャンキング・ベクトル化してDBに挿入"""
    if not database.db_client or not settings.settings_manager or not document_processor.simple_processor:
        raise HTTPException(503, "システムが初期化されていません")
    
    try:
        filename = file.filename
        content = await file.read()
        
        logging.info(f"ファイルアップロード受信: {filename} (カテゴリ: {category}, モデル: {embedding_model})")

        # 1. 古いデータを削除
        logging.info(f"古いチャンク (source: {filename}) を削除しています...")
        try:
            database.db_client.client.table("documents").delete().eq("metadata->>source", filename).execute()
        except Exception as e:
            logging.warning(f"古いチャンク削除時の軽微なエラー (無視可): {e}")

        # 2. ジェネレータを取得 (ここではまだチャンク生成は始まらない)
        collection_name = settings.settings_manager.settings.get("collection", ACTIVE_COLLECTION_NAME)
        doc_generator = document_processor.simple_processor.process_and_chunk(filename, content, category, collection_name)
        
        if doc_generator is None:
             raise HTTPException(status_code=400, detail="ファイル処理の初期化に失敗しました")

        # 3. ストリーミング・バッチ処理
        #    リストに全データを溜め込まず、少しずつ処理してメモリを解放する
        batch_docs = []
        batch_size = 50 # Render (512MB) 環境では 50程度が安全
        total_count = 0
        
        # ジェネレータから1つずつ取り出す
        for doc in doc_generator:
            batch_docs.append(doc)
            
            # バッチサイズに達したら処理実行
            if len(batch_docs) >= batch_size:
                inserted = await process_batch_insert(batch_docs, embedding_model, collection_name)
                total_count += inserted
                batch_docs = [] # メモリ解放！
                await asyncio.sleep(0.5) # APIレート制限対策

        # 端数（ループ終了後に残っている分）を処理
        if batch_docs:
            inserted = await process_batch_insert(batch_docs, embedding_model, collection_name)
            total_count += inserted

        logging.info(f"ファイル処理完了: {filename} (合計 {total_count} 件のチャンクをDBに挿入)")
        return {"chunks": total_count, "filename": filename, "message": "処理完了"}

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"ファイルアップロード処理エラー: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scrape")
async def scrape_website(
    request: ScrapeRequest, 
    user: dict = Depends(require_auth)
):
    """
    URLからHTMLを取得し、Gemini APIを使用して本文テキストのみを賢く抽出します。
    その後、ベクトル化してDBに保存します。
    """
    if not database.db_client or not settings.settings_manager or not document_processor.simple_processor:
        raise HTTPException(503, "システムが初期化されていません")

    logging.info(f"AI Scrapeリクエスト受信: {request.url} (Collection: {request.collection_name})")

    try:
        # 1. WebサイトからHTMLを取得 (httpxを使用)
        # ※ブラウザのふりをするためにUser-Agentを設定
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        async with httpx.AsyncClient(verify=False, headers=headers) as client:
            try:
                response = await client.get(request.url, follow_redirects=True, timeout=15.0)
                response.raise_for_status()
            except httpx.RequestError as e:
                logging.error(f"URL取得エラー: {e}")
                raise HTTPException(status_code=400, detail=f"URLの取得に失敗しました: {e}")
        
        # 2. HTMLの軽量化 (トークン節約のためBeautifulSoupで最低限のゴミ掃除)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 明らかに不要なタグを削除
        for element in soup(["script", "style", "noscript", "iframe", "svg", "header", "footer"]):
            element.decompose()
            
        # HTMLを文字列化（長すぎる場合はカットする安全策を入れる例）
        # gemini-1.5-flash は100万トークン扱えるので基本的にはそのままでOKですが、
        # 極端に巨大なページ対策としてbodyのみに絞ります。
        target_html = str(soup.body) if soup.body else str(soup)
        
        # 3. Gemini API を叩いて本文抽出 (ここが変更点)
        logging.info("GeminiによるHTML解析を実行中...")
        
        # 高速な Flash モデルを指定
        extract_model = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = """
        以下のHTMLソースコードから、Webページの「メインコンテンツ（本文）」のみを抽出してください。
        
        # 指示:
        - メニュー、ナビゲーション、広告、著作権表示、サイドバーのリンク集などは全て除外してください。
        - 記事のタイトル、見出し、本文の段落構造は維持してください。
        - 出力はMarkdown形式ではなく、読みやすいプレーンテキストにしてください。
        - HTMLタグは残さないでください。
        
        HTMLソース:
        """
        
        try:
            # HTMLとプロンプトを送信
            ai_response = await extract_model.generate_content_async(
                [prompt, target_html],
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            body_text = ai_response.text
            
        except Exception as e:
            logging.error(f"Gemini解析エラー: {e}")
            # エラー時はフォールバックとして従来の単純抽出を行う
            logging.warning("Gemini解析に失敗したため、従来の抽出処理を行います。")
            body_text = soup.get_text(separator=' ', strip=True)

        if not body_text:
            raise HTTPException(status_code=400, detail="テキストを抽出できませんでした。")
            
        logging.info(f"抽出されたテキストサイズ: {len(body_text)} 文字")

        # 4. ファイル名定義と古いデータの削除
        filename_from_url = request.url.split('/')[-1] or "index.html"
        # URLパラメータなどを削除してクリーンなファイル名にする簡易処理
        filename_from_url = filename_from_url.split('?')[0] 
        source_name = f"scrape_{filename_from_url}.txt"

        logging.info(f"古いチャンク (source: {source_name}) を削除しています...")
        try:
            database.db_client.client.table("documents").delete().eq("metadata->>source", source_name).execute()
        except Exception as e:
            logging.warning(f"古いチャンク削除時の軽微なエラー: {e}")

        # 5. ジェネレータを取得 (既存ロジック)
        doc_generator = document_processor.simple_processor.process_and_chunk(
            filename=source_name, 
            content=body_text.encode('utf-8'), 
            category="WebScrape", 
            collection_name=request.collection_name
        )
        
        # 6. ストリーミング・バッチ処理 (既存ロジック)
        batch_docs = []
        batch_size = 50
        total_count = 0
        embedding_model = request.embedding_model

        for doc in doc_generator:
            batch_docs.append(doc)
            
            if len(batch_docs) >= batch_size:
                inserted = await process_batch_insert(batch_docs, embedding_model, request.collection_name)
                total_count += inserted
                batch_docs = [] # メモリ解放
                await asyncio.sleep(0.5)

        # 端数処理
        if batch_docs:
            inserted = await process_batch_insert(batch_docs, embedding_model, request.collection_name)
            total_count += inserted

        logging.info(f"スクレイプ処理完了: {request.url} (合計 {total_count} 件のチャンクをDBに挿入)")
        return {"chunks": total_count, "filename": request.url, "message": "処理完了", "preview": body_text[:200] + "..."}

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"スクレイプ処理エラー: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{collection_name}/documents")
async def get_documents(collection_name: str):
    if not database.db_client:
        raise HTTPException(503, "DB not initialized")
    return {
        "documents": database.db_client.get_documents_by_collection(collection_name),
        "count": database.db_client.count_chunks_in_collection(collection_name)
    }