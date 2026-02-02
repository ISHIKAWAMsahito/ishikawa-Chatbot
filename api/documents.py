import logging
import asyncio
import traceback
import json
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Query
import google.generativeai as genai
import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel

# リクエストモデル
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

# ----------------------------------------------------------------
# 共通設定: 整形ルール
# ----------------------------------------------------------------
COMMON_CLEANING_INSTRUCTION = """
【整形ルール】
1. テキストは読みやすいMarkdown形式に整形してください。
2. カレンダーや行事予定表は「YYYY年M月D日(曜): 内容」の形式に統一してください。
3. 無意味な記号、装飾、ヘッダー/フッターの繰り返しは削除してください。
4. 丸数字（①など）は標準数字に変換してください。
"""

# ----------------------------------------------------------------
# ★品質重視のヘルパー関数: 親子チャンキング対応バッチ保存
# ----------------------------------------------------------------
async def process_batch_insert(batch_docs: List[Any], embedding_model: str, collection_name: str):
    """
    【親子チャンキングの中核ロジック】
    - ベクトル化 (Embedding): 「子チャンク (page_content)」を使用。検索の具体性を高めるため。
    - DB保存 (Insert): 「親チャンク (parent_content)」を使用。AIに文脈を理解させるため。
    """
    if not batch_docs:
        return 0

    # 1. ベクトル化対象は「子チャンク（短い文）」
    batch_texts = [doc.page_content for doc in batch_docs]
    max_retries = 3
    
    for attempt in range(max_retries + 1):
        try:
            # Gemini Embedding API 呼び出し
            embedding_response = genai.embed_content(
                model=embedding_model,
                content=batch_texts,
                task_type="retrieval_document"
            )
            embeddings = embedding_response["embedding"]
            
            inserted_count = 0
            for j, doc in enumerate(batch_docs):
                # コレクション名の補完
                if "collection_name" not in doc.metadata:
                    doc.metadata["collection_name"] = collection_name
                
                # --- ★ここが品質向上のキモ ---
                # メタデータに親コンテンツがあれば、それをメインの content に採用
                # なければ（短い文書などの場合）、子コンテンツをそのまま使う
                final_content = doc.metadata.get("parent_content")
                if not final_content:
                    # document_processorの旧バージョン対応
                    final_content = doc.metadata.get("parent_context", doc.page_content)
                
                # デバッグ・リランク精度向上のため、子チャンクの内容もメタデータに記録
                doc.metadata["child_content"] = doc.page_content
                
                # DB容量削減: contentカラムに保存する親テキストは、メタデータからは削除しても良い
                # (ただし、念のため残しておきたい場合はこの delete 行をコメントアウト)
                if "parent_content" in doc.metadata:
                    del doc.metadata["parent_content"]
                if "parent_context" in doc.metadata:
                    del doc.metadata["parent_context"]

                # DBへの保存
                database.db_client.insert_document(
                    content=final_content,  # AIが読むのは「親（文脈あり）」
                    embedding=embeddings[j], # 検索されるのは「子（具体的）」のベクトル
                    metadata=doc.metadata
                )
                inserted_count += 1
                
            return inserted_count

        except Exception as e:
            # レート制限(429)時のリトライ処理
            is_quota_error = "429" in str(e) or "quota" in str(e).lower()
            if is_quota_error and attempt < max_retries:
                wait_time = 60
                logging.warning(f"API制限(429)検知。{wait_time}秒待機してリトライ... ({attempt+1}/{max_retries})")
                await asyncio.sleep(wait_time)
                continue
            else:
                logging.error(f"バッチ処理エラー: {e}")
                raise e

# ----------------------------------------------------------------
# ファイルアップロード処理 (親子チャンキング + 画像パス引継ぎ)
# ----------------------------------------------------------------
@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...), 
    category: str = Form("その他"), 
    embedding_model: str = Form("models/gemini-embedding-001"), 
    user: dict = Depends(require_auth)
):
    if not database.db_client or not document_processor.simple_processor:
        raise HTTPException(503, "システム初期化エラー: DBまたはプロセッサが準備できていません")
    
    try:
        filename = file.filename
        content = await file.read()
        mime_type = file.content_type
        logging.info(f"Upload Start: {filename} ({mime_type})")

        # 1. 画像パス引継ぎのための事前チェック
        preserved_image_path = None
        try:
            # 既存データの source が一致するものを検索
            res = database.db_client.client.table("documents") \
                .select("metadata") \
                .eq("metadata->>source", filename) \
                .limit(1).execute()
            
            if res.data:
                meta = res.data[0].get('metadata', {})
                if isinstance(meta, str): meta = json.loads(meta)
                preserved_image_path = meta.get('image_path')
                if preserved_image_path:
                    logging.info(f"✨ 既存の画像リンク引継ぎ: {preserved_image_path}")
        except Exception as e:
            logging.warning(f"メタデータ確認中の軽微なエラー: {e}")

        # 2. 古いデータの削除 (更新のため)
        try:
            database.db_client.client.table("documents").delete().eq("metadata->>source", filename).execute()
        except Exception:
            pass

        # 3. Gemini Flash によるOCR・整形
        extract_model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"以下のファイルを読み取り、テキストを抽出してください。\n{COMMON_CLEANING_INSTRUCTION}"
        
        try:
            ai_response = await extract_model.generate_content_async(
                [prompt, {"mime_type": mime_type, "data": content}]
            )
            cleaned_text = ai_response.text
        except Exception as e:
            logging.error(f"Gemini OCRエラー: {e}")
            # テキストファイルならそのまま使うフォールバック
            if "text" in mime_type:
                cleaned_text = content.decode("utf-8", errors="ignore")
            else:
                raise HTTPException(500, "AIによるファイル読み取りに失敗しました")

        # 4. document_processor による親子チャンキング
        collection_name = settings.settings_manager.settings.get("collection", ACTIVE_COLLECTION_NAME)
        
        # ファイル名を内部処理用に整える
        proc_filename = filename if filename.endswith(('.txt', '.md', '.csv')) else f"{filename}.txt"

        doc_generator = document_processor.simple_processor.process_and_chunk(
            filename=proc_filename, 
            content=cleaned_text.encode('utf-8'), 
            category=category, 
            collection_name=collection_name
        )
        
        if doc_generator is None:
             raise HTTPException(400, "ファイル処理の初期化に失敗しました")

        # 5. バッチ保存処理
        batch_docs = []
        total_chunks = 0
        
        for doc in doc_generator:
            doc.metadata["source"] = filename 
            # 画像パスがあれば再注入
            if preserved_image_path:
                doc.metadata["image_path"] = preserved_image_path

            batch_docs.append(doc)
            
            # 15件ごとにまとめてAPIへ（レート制限回避のため少し待機）
            if len(batch_docs) >= 15:
                inserted = await process_batch_insert(batch_docs, embedding_model, collection_name)
                total_chunks += inserted
                batch_docs = []
                await asyncio.sleep(0.5)

        # 残りの端数を保存
        if batch_docs:
            inserted = await process_batch_insert(batch_docs, embedding_model, collection_name)
            total_chunks += inserted

        return {
            "chunks": total_chunks, 
            "filename": filename, 
            "message": "完了: 親子チャンキングにより高精度な検索データが構築されました"
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"処理エラー: {str(e)}")

# ----------------------------------------------------------------
# Webスクレイピング処理 (親子チャンキング対応)
# ----------------------------------------------------------------
@router.post("/scrape")
async def scrape_website(request: ScrapeRequest, user: dict = Depends(require_auth)):
    if not database.db_client or not document_processor.simple_processor:
        raise HTTPException(503, "システム初期化エラー")

    logging.info(f"Scrape Request: {request.url}")

    try:
        # 1. サイトへのアクセス
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        async with httpx.AsyncClient(verify=False, headers=headers, follow_redirects=True) as client:
            resp = await client.get(request.url, timeout=30.0)
            resp.raise_for_status()
            content_body = resp.content

        # 2. PDF判定
        is_pdf = content_body.startswith(b'%PDF') or "application/pdf" in resp.headers.get("content-type", "").lower()
        target_type = "PDF" if is_pdf else "HTML"

        # 3. Gemini Flash による抽出・整形
        extract_model = genai.GenerativeModel("gemini-2.5-flash")
        
        if target_type == "PDF":
            prompt = f"このPDFのテキストを抽出・整形してください。\n{COMMON_CLEANING_INSTRUCTION}"
            ai_data = {"mime_type": "application/pdf", "data": content_body}
        else:
            # HTMLの場合、余計なタグを消してからAIへ
            soup = BeautifulSoup(resp.text, 'html.parser')
            for tag in soup(["script", "style", "nav", "footer"]): tag.decompose()
            main_html = str(soup.body) if soup.body else str(soup)
            prompt = f"以下のHTMLから本文を抽出・整形してください。\n{COMMON_CLEANING_INSTRUCTION}"
            ai_data = main_html

        ai_resp = await extract_model.generate_content_async([prompt, ai_data])
        extracted_text = ai_resp.text

        # 4. 保存準備 (画像パス引継ぎなど)
        url_filename = request.url.split('/')[-1].split('?')[0] or "index.html"
        source_name = f"scrape_{url_filename}"
        
        # 既存画像パスの救出
        preserved_image_path = None
        try:
            res = database.db_client.client.table("documents").select("metadata").eq("metadata->>source", source_name).limit(1).execute()
            if res.data:
                meta = res.data[0].get('metadata', {})
                if isinstance(meta, str): meta = json.loads(meta)
                preserved_image_path = meta.get('image_path')
        except Exception: pass

        # 古いデータ削除
        database.db_client.client.table("documents").delete().eq("metadata->>source", source_name).execute()

        # 5. 親子チャンキング処理
        doc_generator = document_processor.simple_processor.process_and_chunk(
            filename=source_name + ".txt", 
            content=extracted_text.encode('utf-8'), 
            category=f"WebScrape({target_type})", 
            collection_name=request.collection_name
        )

        batch_docs = []
        total_chunks = 0
        for doc in doc_generator:
            doc.metadata["url"] = request.url
            doc.metadata["source"] = source_name
            if preserved_image_path:
                doc.metadata["image_path"] = preserved_image_path

            batch_docs.append(doc)
            if len(batch_docs) >= 50:
                inserted = await process_batch_insert(batch_docs, request.embedding_model, request.collection_name)
                total_chunks += inserted
                batch_docs = []
                await asyncio.sleep(0.5)

        if batch_docs:
            total_chunks += await process_batch_insert(batch_docs, request.embedding_model, request.collection_name)

        return {
            "chunks": total_chunks, 
            "message": "Webサイト情報の取り込み完了（親子チャンキング適用）",
            "source": source_name
        }

    except Exception as e:
        logging.error(f"Scrape error: {traceback.format_exc()}")
        raise HTTPException(500, f"スクレイピング失敗: {str(e)}")

# ----------------------------------------------------------------
# DB管理画面 (DB.html) 用のエンドポイント群
# ----------------------------------------------------------------

@router.get("/all")
async def get_all_documents(
    page: int = Query(1, ge=1),
    limit: int = Query(100, ge=1),
    search: Optional[str] = None,
    category: Optional[str] = None
):
    """
    DB管理画面用：すべてのドキュメントをページネーション付きで取得
    """
    if not database.db_client:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        # 1. クエリの基本形を作成 (総件数取得のため count="exact" を指定)
        query = database.db_client.client.table("documents").select("*", count="exact")

        # 2. 検索・フィルタリング
        if search:
            # コンテンツ内容またはメタデータのソース名から部分一致検索
            query = query.or_(f"content.ilike.%{search}%,metadata->>source.ilike.%{search}%")
        
        if category:
            query = query.eq("metadata->>category", category)

        # 3. ページネーションの範囲計算
        start = (page - 1) * limit
        end = start + limit - 1

        # 4. 実行 (最新順にソート)
        response = query.order("created_at", desc=True).range(start, end).execute()

        return {
            "documents": response.data,
            "total": response.count,
            "page": page,
            "limit": limit
        }

    except Exception as e:
        logging.error(f"Error fetching documents: {e}")
        raise HTTPException(status_code=500, detail="データ取得に失敗しました")

@router.get("/{id}")
async def get_document_by_id(id: int):
    """
    指定されたIDのドキュメントを1件取得 (編集モーダル用)
    """
    try:
        res = database.db_client.client.table("documents").select("*").eq("id", id).single().execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Document not found")
        return res.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{id}")
async def update_document(id: int, data: Dict[str, Any]):
    """
    レコードの編集内容を保存
    """
    try:
        # DB.html から送られてくる content と metadata を更新
        database.db_client.client.table("documents").update({
            "content": data.get("content"),
            "metadata": data.get("metadata")
        }).eq("id", id).execute()
        return {"message": "Updated successfully"}
    except Exception as e:
        logging.error(f"Update error: {e}")
        raise HTTPException(status_code=500, detail="更新に失敗しました")

@router.delete("/{id}")
async def delete_document(id: int):
    """
    指定されたレコードを個別に削除
    """
    try:
        database.db_client.client.table("documents").delete().eq("id", id).execute()
        return {"message": "Deleted successfully"}
    except Exception as e:
        logging.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail="削除に失敗しました")