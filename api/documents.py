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


# api/documents.py

@router.post("/scrape")#12/24 「ダウンロードしたデータが本当にPDFなのか（%PDFで始まっているか）を厳密にチェックし、ダメなら処理を中断して理由を教えてくれる」 安全装置付きのコードに書き換え
async def scrape_website(
    request: ScrapeRequest, 
    user: dict = Depends(require_auth)
):
    """
    URLからコンテンツを取得し、PDFかどうか厳密に判定してからGeminiで解析します。
    大学サイトなどのBot対策(403/Blocked)を回避するためのヘッダー強化版です。
    """
    if not database.db_client or not settings.settings_manager or not document_processor.simple_processor:
        raise HTTPException(503, "システムが初期化されていません")

    logging.info(f"AI Scrapeリクエスト受信: {request.url}")

    try:
        # 1. 人間のブラウザになりすましてアクセス (大学サイト対策)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.google.com/",  # Google検索から来たふりをする
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }

        async with httpx.AsyncClient(verify=False, headers=headers, follow_redirects=True) as client:
            try:
                response = await client.get(request.url, timeout=30.0)
                response.raise_for_status() # 404や500ならここでエラー
                content_body = response.content
            except httpx.RequestError as e:
                logging.error(f"接続エラー: {e}")
                raise HTTPException(status_code=400, detail=f"URLに接続できませんでした: {e}")
            except httpx.HTTPStatusError as e:
                logging.error(f"HTTPエラー: {e.response.status_code}")
                raise HTTPException(status_code=400, detail=f"Webサイトへのアクセスが拒否されました(Status: {e.response.status_code})")

        # 2. 【最重要】中身が本当にPDFかチェックする
        # PDFファイルは必ずバイナリの先頭が "%PDF" で始まります。
        # 始まっていない場合、それは「アクセス拒否画面のHTML」である可能性が高いです。
        
        is_pdf_signature = content_body.startswith(b'%PDF')
        content_type_header = response.headers.get("content-type", "").lower()
        
        # ログに先頭50バイトを出力して確認できるようにする
        file_head_preview = content_body[:50].decode('utf-8', errors='ignore').replace('\n', ' ')
        logging.info(f"ダウンロードデータの先頭: {file_head_preview}")

        # PDF判定ロジック
        if is_pdf_signature:
            # マジックナンバーが合致すれば確実にPDF
            target_type = "PDF"
        elif "application/pdf" in content_type_header:
            # ヘッダーはPDFだが中身が違う -> 破損かブロック
            logging.error(f"ヘッダーはPDFですが、データ構造がPDFではありません。先頭データ: {file_head_preview}")
            raise HTTPException(status_code=400, detail="PDFとしてダウンロードされましたが、ファイルが破損しているか、アクセス拒否ページの可能性があります。")
        else:
            # それ以外はHTMLとして扱う
            target_type = "HTML"

        extracted_text = ""
        extract_model = genai.GenerativeModel("gemini-1.5-flash")

        # ---------------------------------------------------------
        # 分岐 A: PDF処理
        # ---------------------------------------------------------
        if target_type == "PDF":
            logging.info(f"有効なPDFファイルを検知しました ({len(content_body)} bytes)。解析を開始します。")
            
            prompt = """
            このPDFファイルの内容を読み取り、全てのテキストを抽出してください。
            図表が含まれている場合は、その内容も言葉で説明してテキストに含めてください。
            ヘッダーやフッター（ページ番号など）は除外してください。
            """
            try:
                ai_response = await extract_model.generate_content_async(
                    [prompt, {"mime_type": "application/pdf", "data": content_body}]
                )
                extracted_text = ai_response.text
            except Exception as e:
                logging.error(f"Gemini解析エラー(PDF): {e}")
                # AI側でエラーが出た場合の詳細
                raise HTTPException(status_code=500, detail=f"AIによるPDF解析に失敗しました。ファイルが重すぎるか、保護されています。: {str(e)}")

        # ---------------------------------------------------------
        # 分岐 B: HTML処理
        # ---------------------------------------------------------
        else:
            logging.info("HTMLコンテンツとして解析します。")
            
            # HTMLなのに中身が空っぽ、あるいは短すぎる場合のエラーチェック
            if len(content_body) < 100:
                raise HTTPException(status_code=400, detail="取得したWebページの内容が短すぎます（アクセスがブロックされた可能性があります）。")

            soup = BeautifulSoup(response.text, 'html.parser')
            for element in soup(["script", "style", "noscript", "iframe", "svg", "header", "footer"]):
                element.decompose()
            
            target_html = str(soup.body) if soup.body else str(soup)
            
            prompt = "以下のHTMLから本文テキストのみを抽出してください（メニュー等は除外）。"
            
            try:
                ai_response = await extract_model.generate_content_async([prompt, target_html])
                extracted_text = ai_response.text
            except Exception as e:
                logging.warning(f"AI解析失敗、フォールバック: {e}")
                extracted_text = soup.get_text(separator=' ', strip=True)

        if not extracted_text:
            raise HTTPException(status_code=400, detail="テキストを抽出できませんでした。")

        # ---------------------------------------------------------
        # 以下、保存処理 (そのまま)
        # ---------------------------------------------------------
        filename_from_url = request.url.split('/')[-1].split('?')[0] or "downloaded_file"
        if target_type == "PDF" and not filename_from_url.endswith(".pdf"):
            filename_from_url += ".pdf"
        elif target_type == "HTML" and not filename_from_url.endswith((".html", ".txt")):
             filename_from_url += ".txt"
        
        source_name = f"scrape_{filename_from_url}"

        # 古いデータの削除
        try:
            database.db_client.client.table("documents").delete().eq("metadata->>source", source_name).execute()
        except Exception:
            pass

        # チャンク化と保存
        doc_generator = document_processor.simple_processor.process_and_chunk(
            filename=source_name, 
            content=extracted_text.encode('utf-8'), 
            category=f"WebScrape({target_type})", 
            collection_name=request.collection_name
        )
        
        batch_docs = []
        total_count = 0
        
        for doc in doc_generator:
            batch_docs.append(doc)
            if len(batch_docs) >= 50:
                total_count += await process_batch_insert(batch_docs, request.embedding_model, request.collection_name)
                batch_docs = []
                await asyncio.sleep(0.5)
        
        if batch_docs:
            total_count += await process_batch_insert(batch_docs, request.embedding_model, request.collection_name)

        return {
            "chunks": total_count, 
            "filename": filename_from_url, 
            "message": "処理完了", 
            "type": target_type
        }

    except HTTPException:
        raise
    except Exception as e:
        # 想定外のクラッシュも500エラーとして詳細を返す
        logging.error(f"システムエラー: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"システム内部エラー: {str(e)}")


@router.get("/collections/{collection_name}/documents")
async def get_documents(collection_name: str):
    if not database.db_client:
        raise HTTPException(503, "DB not initialized")
    return {
        "documents": database.db_client.get_documents_by_collection(collection_name),
        "count": database.db_client.count_chunks_in_collection(collection_name)
    }