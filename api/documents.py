import logging
import asyncio
import traceback
import json
import hashlib
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
                if "parent_content" in doc.metadata:
                     # メタデータからは削除（DB容量節約のため）
                    del doc.metadata["parent_content"]

                if not final_content:
                    # document_processorの旧バージョン対応
                    final_content = doc.metadata.get("parent_context", doc.page_content)
                    if "parent_context" in doc.metadata:
                        del doc.metadata["parent_context"]
                
                # デバッグ・リランク精度向上のため、子チャンクの内容もメタデータに記録
                doc.metadata["child_content"] = doc.page_content
                
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

    try:
        # 1. サイトへのアクセス
        headers = {"User-Agent": "Mozilla/5.0 ..."} # 既存のヘッダー
        async with httpx.AsyncClient(verify=False, headers=headers, follow_redirects=True) as client:
            resp = await client.get(request.url, timeout=30.0)
            resp.raise_for_status()
            content_body = resp.content

        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # --- ★ 独自ロジックの注入: contactBox等の事前抽出 ---
        output_blocks = []
        # 連絡先ボックスの解析 (提示いただいたコードを最適化して挿入)
        contact_main = soup.find('div', class_='contactBox__main')
        if contact_main:
            desc_p = contact_main.find('p', class_='contactBox__desc')
            if desc_p: output_blocks.append(f"【窓口説明】: {desc_p.get_text(strip=True)}")
            
            office_div = contact_main.find('div', class_='contactBox__office')
            if office_div:
                title = office_div.find('p', class_='title')
                phone = office_div.find('dl', class_='phone')
                if title and phone:
                    output_blocks.append(f"【部署名】: {title.get_text(strip=True)}")
                    output_blocks.append(f"【電話番号】: {phone.get_text(strip=True)}")

        # --- ★ 重複防止策: URLに基づく一意のソース名生成 ---
        # URLを元にした固定のIDを作ることで、職員が何度ボタンを押しても「上書き」になるようにする
        url_hash = hashlib.md5(request.url.encode()).hexdigest()[:8]
        page_title = soup.title.string.strip() if soup.title else "名称未設定"
        source_name = f"scrape_{page_title}_{url_hash}"

        # 不要なタグを消してメイン本文も取得
        for tag in soup(["script", "style", "nav", "footer", "iframe"]): tag.decompose()
        main_html = str(soup.body) if soup.body else str(soup)

        # 2. Geminiによる整形 (抽出したcontactBox情報をヒントとして渡す)
        extract_model = genai.GenerativeModel("gemini-2.5-flash")
        extracted_info = "\n".join(output_blocks)
        prompt = f"""
        以下のHTMLから本文を抽出してください。
        特に以下の情報は重要なので、正確に含めてください:
        {extracted_info}
        {COMMON_CLEANING_INSTRUCTION}
        """
        ai_resp = await extract_model.generate_content_async([prompt, main_html])
        cleaned_text = ai_resp.text

        # 3. 古いデータの削除 (source_nameが一致するものを全削除してから入れ直す)
        database.db_client.client.table("documents").delete().eq("metadata->>source", source_name).execute()

        # 4. 親子チャンキング保存
        doc_generator = document_processor.simple_processor.process_and_chunk(
            filename=source_name, 
            content=cleaned_text.encode('utf-8'), 
            category="Webスクレイピング", 
            collection_name=request.collection_name
        )

        # 重複チャンクの排除（同じ親の中に同じ文章が生成された場合のガード）
        seen_contents = set()
        batch_docs = []
        total_chunks = 0

        for doc in doc_generator:
            # 完全に同じ内容のチャンクがあればスキップ
            chunk_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if chunk_hash in seen_contents:
                continue
            seen_contents.add(chunk_hash)

            doc.metadata.update({"url": request.url, "source": source_name})
            batch_docs.append(doc)
            
            if len(batch_docs) >= 30:
                total_chunks += await process_batch_insert(batch_docs, request.embedding_model, request.collection_name)
                batch_docs = []
                
        if batch_docs:
            total_chunks += await process_batch_insert(batch_docs, request.embedding_model, request.collection_name)

        return {"message": f"「{page_title}」の取り込みが完了しました", "chunks": total_chunks}

    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(500, f"エラーが発生しました: {str(e)}")

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
        # 1. クライアントの安全な取得 (main.pyと同様のロジック)
        client = database.db_client
        if hasattr(client, "client"):
            client = client.client
        
        # 2. クエリの基本形を作成
        query = client.table("documents").select("*", count="exact")

        # 3. 検索・フィルタリング
        if search:
            # ilikeフィルタの構文エラーを防ぐため、単純な文字列であることを想定
            clean_search = search.replace(",", "").replace("%", "") # 簡易サニタイズ
            query = query.or_(f"content.ilike.%{clean_search}%,metadata->>source.ilike.%{clean_search}%")
        
        if category:
            query = query.eq("metadata->>category", category)

        # 4. ページネーションの範囲計算
        start = (page - 1) * limit
        end = start + limit - 1

        # 5. 実行
        # 【修正】 created_at がテーブルに存在しないため、id でソートするように変更
        response = query.order("id", desc=True).range(start, end).execute()

        return {
            "documents": response.data,
            "total": response.count,
            "page": page,
            "limit": limit
        }

    except Exception as e:
        # ログに詳細を出力し、フロントエンドにもエラー内容を返す
        error_msg = str(e)
        logging.error(f"Error fetching documents: {traceback.format_exc()}")
        
        raise HTTPException(status_code=500, detail=f"データ取得エラー: {error_msg}")
# documents.py の末尾などに追加

@router.get("/collections/{collection_name}/documents")
async def get_collection_documents(
    collection_name: str,
    page: int = Query(1, ge=1),
    limit: int = Query(100, ge=1),
    search: Optional[str] = None
):
    """
    フロントエンドの要求に合わせて、特定のコレクションに属するドキュメントを取得する
    """
    if not database.db_client:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        client = database.db_client.client if hasattr(database.db_client, "client") else database.db_client
        
        # 1. 基本クエリ（メタデータ内の collection_name でフィルタリング）
        # ※ スクレイピング時に metadata に collection_name を入れている仕様に合わせる
        query = client.table("documents").select("*", count="exact") \
                      .eq("metadata->>collection_name", collection_name)

        # 2. 検索キーワードがある場合
        if search:
            query = query.or_(f"content.ilike.%{search}%,metadata->>source.ilike.%{search}%")

        # 3. ページネーション設定
        start = (page - 1) * limit
        end = start + limit - 1

        # 4. 実行（ID順にソート）
        response = query.order("id", desc=True).range(start, end).execute()

        return {
            "documents": response.data,
            "total": response.count,
            "page": page,
            "limit": limit
        }
    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
# documents.py に追加

@router.get("/{document_id}")
async def get_document_by_id(document_id: int):
    """IDを指定して単一のドキュメントを取得する"""
    if not database.db_client:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    client = database.db_client.client if hasattr(database.db_client, "client") else database.db_client
    response = client.table("documents").select("*").eq("id", document_id).single().execute()
    
    if not response.data:
        raise HTTPException(status_code=404, detail="Document not found")
        
    return response.data

@router.delete("/{document_id}")
async def delete_document(document_id: int, user: dict = Depends(require_auth)):
    """IDを指定してドキュメントを削除する"""
    if not database.db_client:
        raise HTTPException(status_code=503, detail="Database not initialized")
        
    client = database.db_client.client if hasattr(database.db_client, "client") else database.db_client
    try:
        client.table("documents").delete().eq("id", document_id).execute()
        return {"message": f"Document {document_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))