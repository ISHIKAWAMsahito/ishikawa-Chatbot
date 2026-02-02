import logging
import asyncio
import traceback
import json
import hashlib
import re
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
# 共通設定: 整形ルール (Markdown化を強制)
# ----------------------------------------------------------------
COMMON_CLEANING_INSTRUCTION = """
【整形ルール】
1. HTMLの構造（h1, h2, ul, liなど）を維持し、適切なMarkdown形式（# 見出し, - リスト）に変換してください。
2. リンクは `[リンクテキスト](URL)` の形式にしてください。
3. カレンダーや行事予定表は「YYYY年M月D日(曜): 内容」の形式に統一してください。
4. 無意味な装飾やナビゲーションメニューは削除し、本文のみを抽出してください。
"""

# ----------------------------------------------------------------
# ★品質重視のヘルパー関数: 親子チャンキング対応バッチ保存
# ----------------------------------------------------------------
async def process_batch_insert(batch_docs: List[Any], embedding_model: str, collection_name: str):
    if not batch_docs:
        return 0

    batch_texts = [doc.page_content for doc in batch_docs]
    max_retries = 3
    
    for attempt in range(max_retries + 1):
        try:
            embedding_response = genai.embed_content(
                model=embedding_model,
                content=batch_texts,
                task_type="retrieval_document"
            )
            embeddings = embedding_response["embedding"]
            
            inserted_count = 0
            for j, doc in enumerate(batch_docs):
                if "collection_name" not in doc.metadata:
                    doc.metadata["collection_name"] = collection_name
                
                # 親コンテンツの取得ロジック
                final_content = doc.metadata.get("parent_content")
                if "parent_content" in doc.metadata:
                    del doc.metadata["parent_content"]

                if not final_content:
                    final_content = doc.metadata.get("parent_context", doc.page_content)
                    if "parent_context" in doc.metadata:
                        del doc.metadata["parent_context"]
                
                doc.metadata["child_content"] = doc.page_content
                
                database.db_client.insert_document(
                    content=final_content, 
                    embedding=embeddings[j],
                    metadata=doc.metadata
                )
                inserted_count += 1
                
            return inserted_count

        except Exception as e:
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
# Webスクレイピング処理 (Markdown保存対応版)
# ----------------------------------------------------------------
@router.post("/scrape")
async def scrape_website(request: ScrapeRequest, user: dict = Depends(require_auth)):
    if not database.db_client or not document_processor.simple_processor:
        raise HTTPException(503, "システム初期化エラー")

    try:
        # 1. サイトへのアクセス
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        async with httpx.AsyncClient(verify=False, headers=headers, follow_redirects=True) as client:
            resp = await client.get(request.url, timeout=30.0)
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # --- ★ 独自ロジック: 連絡先情報の事前抽出 ---
        output_blocks = []
        contact_box_main = soup.find('div', class_='contactBox__main')
        if contact_box_main:
            desc_p = contact_box_main.find('p', class_='contactBox__desc')
            if desc_p:
                output_blocks.append(f"【窓口説明】: {desc_p.get_text(strip=True)}")
            
            office_div = contact_box_main.find('div', class_='contactBox__office')
            if office_div:
                title_p = office_div.find('p', class_='title')
                if title_p:
                    output_blocks.append(f"【部署名】: {title_p.get_text(strip=True)}")
                phone_dl = office_div.find('dl', class_='phone')
                if phone_dl:
                    output_blocks.append(f"【電話番号】: {phone_dl.get_text(strip=True)}")

        # --- ★ Markdownファイル名生成ロジック ---
        # URLハッシュ化
        url_hash = hashlib.md5(request.url.encode()).hexdigest()[:8]
        # ページタイトル取得 & ファイル名に使えない文字を除去
        raw_title = soup.title.string.strip() if soup.title else "名称未設定"
        safe_title = re.sub(r'[\\/:*?"<>|]', '', raw_title)
        
        # ★変更点: 拡張子を .md に設定
        source_name = f"scrape_{safe_title}_{url_hash}.md"

        # 不要なタグを消去してHTMLを綺麗にする
        for tag in soup(["script", "style", "nav", "footer", "iframe", "noscript"]): tag.decompose()
        main_html = str(soup.body) if soup.body else str(soup)

        # 2. GeminiによるMarkdown整形
       
        extract_model = genai.GenerativeModel("gemini-2.5-flash")
        extracted_info = "\n".join(output_blocks)
        prompt = f"""
        以下のHTMLコンテンツを解析し、情報を漏らさず整理されたMarkdownテキストに変換してください。
        
        【重要事項】
        - 以下の連絡先情報が含まれている場合は、必ず目立つように記載してください:
        {extracted_info}
        
        {COMMON_CLEANING_INSTRUCTION}
        """
        ai_resp = await extract_model.generate_content_async([prompt, main_html])
        cleaned_text = ai_resp.text

        # 3. 古いデータの削除 (source_name完全一致で削除)
        database.db_client.client.table("documents").delete().eq("metadata->>source", source_name).execute()

        # 4. 親子チャンキング保存
        # ★重要: ファイル名(source_name)が .md で終わっているため、
        # document_processor はMarkdownとして適切にチャンキング処理を行います。
        doc_generator = document_processor.simple_processor.process_and_chunk(
            filename=source_name, 
            content=cleaned_text.encode('utf-8'), 
            category="Webスクレイピング", 
            collection_name=request.collection_name
        )

        seen_contents = set()
        batch_docs = []
        total_chunks = 0

        for doc in doc_generator:
            # 重複チェック
            chunk_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if chunk_hash in seen_contents:
                continue
            seen_contents.add(chunk_hash)

            doc.metadata.update({"url": request.url, "source": source_name})
            batch_docs.append(doc)
            
            if len(batch_docs) >= 30:
                total_chunks += await process_batch_insert(batch_docs, request.embedding_model, request.collection_name)
                batch_docs = []
                await asyncio.sleep(0.2)
                
        if batch_docs:
            total_chunks += await process_batch_insert(batch_docs, request.embedding_model, request.collection_name)

        return {"message": f"「{raw_title}」の取り込み完了 (.md保存)", "chunks": total_chunks}

    except Exception as e:
        logging.error(f"Scrape Error: {e}")
        raise HTTPException(500, f"スクレイピング失敗: {str(e)}")

# --- 以下、既存のエンドポイント (変更なし) ---
@router.get("/all")
async def get_all_documents(page: int = Query(1, ge=1), limit: int = Query(100, ge=1), search: Optional[str] = None, category: Optional[str] = None):
    if not database.db_client: raise HTTPException(503, "Database not initialized")
    try:
        client = database.db_client.client if hasattr(database.db_client, "client") else database.db_client
        query = client.table("documents").select("*", count="exact")
        if search:
            clean_search = search.replace(",", "").replace("%", "")
            query = query.or_(f"content.ilike.%{clean_search}%,metadata->>source.ilike.%{clean_search}%")
        if category: query = query.eq("metadata->>category", category)
        start = (page - 1) * limit
        end = start + limit - 1
        response = query.order("id", desc=True).range(start, end).execute()
        return {"documents": response.data, "total": response.count, "page": page, "limit": limit}
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"データ取得エラー: {str(e)}")

@router.get("/collections/{collection_name}/documents")
async def get_collection_documents(collection_name: str, page: int = Query(1, ge=1), limit: int = Query(100, ge=1), search: Optional[str] = None):
    try:
        client = database.db_client.client if hasattr(database.db_client, "client") else database.db_client
        query = client.table("documents").select("*", count="exact").eq("metadata->>collection_name", collection_name)
        if search: query = query.or_(f"content.ilike.%{search}%,metadata->>source.ilike.%{search}%")
        start = (page - 1) * limit
        end = start + limit - 1
        response = query.order("id", desc=True).range(start, end).execute()
        return {"documents": response.data, "total": response.count, "page": page, "limit": limit}
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@router.get("/{document_id}")
async def get_document_by_id(document_id: int):
    client = database.db_client.client if hasattr(database.db_client, "client") else database.db_client
    response = client.table("documents").select("*").eq("id", document_id).single().execute()
    if not response.data: raise HTTPException(404, "Document not found")
    return response.data

@router.delete("/{document_id}")
async def delete_document(document_id: int, user: dict = Depends(require_auth)):
    client = database.db_client.client if hasattr(database.db_client, "client") else database.db_client
    try:
        client.table("documents").delete().eq("id", document_id).execute()
        return {"message": f"Document {document_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(500, detail=str(e))