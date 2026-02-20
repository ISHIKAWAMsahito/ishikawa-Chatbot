import logging
import asyncio
import traceback
import json
import hashlib
import re
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Query
import google.generativeai as genai
import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

# リクエストモデル（DoS対策: 全フィールドに max_length）
class ScrapeRequest(BaseModel):
    url: str = Field(..., max_length=2048, description="スクレイピング対象URL")
    collection_name: str = Field(..., max_length=256)
    embedding_model: str = Field(..., max_length=128)

from core.dependencies import require_auth
from core import database
from core import settings
from core.config import ACTIVE_COLLECTION_NAME

# ★修正ポイント1: クラスを直接インポートする
from services.document_processor import SimpleDocumentProcessor

router = APIRouter()

# ★修正ポイント2: プロセッサのインスタンスをここで作成する
simple_processor = SimpleDocumentProcessor()

# エラー時は汎用メッセージを返し、詳細はログのみ (情報開示対策)
GENERIC_ERROR_MSG = "処理に失敗しました。"
LOG_EXC_INFO = True

def _sanitize_search(s: Optional[str]) -> str:
    """検索文字列のサニタイズ (ilike のワイルドカード・インジェクション対策)"""
    if not s or not s.strip():
        return ""
    return s.replace(",", "").replace("%", "").replace("_", "").strip()[:500]

def _is_url_allowed_for_scrape(url: str) -> tuple[bool, str]:
    """スクレイピング許可: スキームは https のみ、プライベート/内部ホストは拒否 (SSRF対策)"""
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "無効なURLです。"
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("https",):
        return False, "スキームは https のみ利用可能です。"
    host = (parsed.hostname or "").lower()
    if not host:
        return False, "ホストが取得できません。"
    # ローカル・プライベート・メタデータ系を拒否
    if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return False, "ローカルホストは許可されていません。"
    if host.startswith("169.254.") or host.startswith("10.") or host.startswith("192.168."):
        return False, "内部ネットワークへのアクセスは許可されていません。"
    if host.startswith("172."):
        try:
            second_octet = int(host.split(".")[1] or "0")
            if 16 <= second_octet <= 31:
                return False, "内部ネットワークへのアクセスは許可されていません。"
        except (ValueError, IndexError):
            pass
    if host == "metadata.google.internal" or ".internal" in host:
        return False, "内部メタデータへのアクセスは許可されていません。"
    return True, ""

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
                
                # 子チャンク（短い固有の文章）をメインコンテンツとする
                child_content = doc.page_content
                
                database.db_client.insert_document(
                    content=child_content,
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
# Webスクレイピング処理
# ----------------------------------------------------------------
@router.post("/scrape")
async def scrape_website(request: ScrapeRequest, user: dict = Depends(require_auth)):
    # ★修正ポイント3: 作成した simple_processor が存在するかチェックする
    if not database.db_client or not simple_processor:
        raise HTTPException(503, "システム初期化エラー")

    ok, err_msg = _is_url_allowed_for_scrape(request.url)
    if not ok:
        raise HTTPException(400, err_msg)

    try:
        # 1. サイトへのアクセス
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        async with httpx.AsyncClient(verify=True, headers=headers, follow_redirects=True) as client:
            resp = await client.get(request.url, timeout=30.0)
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # --- 独自ロジック: 連絡先情報の事前抽出 ---
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

        # --- Markdownファイル名生成ロジック ---
        url_hash = hashlib.sha256(request.url.encode()).hexdigest()[:8]
        raw_title = soup.title.string.strip() if soup.title else "名称未設定"
        safe_title = re.sub(r'[\\/:*?"<>|]', '', raw_title)
        
        source_name = f"scrape_{safe_title}_{url_hash}.md"

        # 不要なタグを消去
        for tag in soup(["script", "style", "nav", "footer", "iframe", "noscript"]): tag.decompose()
        main_html = str(soup.body) if soup.body else str(soup)

        # 2. GeminiによるMarkdown整形
        extract_model = genai.GenerativeModel("models/gemini-2.5-flash")
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

        # 3. 古いデータの削除
        database.db_client.client.table("documents").delete().eq("metadata->>source", source_name).execute()

        # 4. 親子チャンキング保存
        # ★修正ポイント4: ここで simple_processor を呼び出す
        doc_generator = simple_processor.process_and_chunk(
            filename=source_name, 
            content=cleaned_text.encode('utf-8'), 
            category="Webスクレイピング", 
            collection_name=request.collection_name
        )

        seen_contents = set()
        batch_docs = []
        total_chunks = 0

        for doc in doc_generator:
            chunk_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()
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

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Scrape Error: {e}", exc_info=LOG_EXC_INFO)
        raise HTTPException(500, "スクレイピングに失敗しました。")

# --- 以下、既存のエンドポイント ---
@router.get("/all")
async def get_all_documents(
    page: int = Query(1, ge=1),
    limit: int = Query(100, ge=1),
    search: Optional[str] = None,
    category: Optional[str] = None,
    user: dict = Depends(require_auth),
):
    if not database.db_client:
        raise HTTPException(503, "Database not initialized")
    try:
        client = database.db_client.client if hasattr(database.db_client, "client") else database.db_client
        query = client.table("documents").select("*", count="exact")
        clean_search = _sanitize_search(search)
        if clean_search:
            query = query.or_(f"content.ilike.%{clean_search}%,metadata->>source.ilike.%{clean_search}%")
        if category:
            query = query.eq("metadata->>category", category)
        start = (page - 1) * limit
        end = start + limit - 1
        response = query.order("id", desc=True).range(start, end).execute()
        return {"documents": response.data, "total": response.count, "page": page, "limit": limit}
    except Exception as e:
        logging.error(f"Documents list error: {e}", exc_info=LOG_EXC_INFO)
        raise HTTPException(status_code=500, detail=GENERIC_ERROR_MSG)

@router.get("/collections/{collection_name}/documents")
async def get_collection_documents(
    collection_name: str,
    page: int = Query(1, ge=1),
    limit: int = Query(100, ge=1),
    search: Optional[str] = None,
    user: dict = Depends(require_auth),
):
    if not database.db_client:
        raise HTTPException(503, "Database not initialized")
    try:
        client = database.db_client.client if hasattr(database.db_client, "client") else database.db_client
        query = client.table("documents").select("*", count="exact").eq("metadata->>collection_name", collection_name)
        clean_search = _sanitize_search(search)
        if clean_search:
            query = query.or_(f"content.ilike.%{clean_search}%,metadata->>source.ilike.%{clean_search}%")
        start = (page - 1) * limit
        end = start + limit - 1
        response = query.order("id", desc=True).range(start, end).execute()
        return {"documents": response.data, "total": response.count, "page": page, "limit": limit}
    except Exception as e:
        logging.error(f"Collection documents error: {e}", exc_info=LOG_EXC_INFO)
        raise HTTPException(500, detail=GENERIC_ERROR_MSG)

@router.get("/{document_id}")
async def get_document_by_id(document_id: int, user: dict = Depends(require_auth)):
    if not database.db_client:
        raise HTTPException(503, "Database not initialized")
    try:
        client = database.db_client.client if hasattr(database.db_client, "client") else database.db_client
        response = client.table("documents").select("*").eq("id", document_id).single().execute()
        if not response.data:
            raise HTTPException(404, "Document not found")
        return response.data
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Document get error: {e}", exc_info=LOG_EXC_INFO)
        raise HTTPException(500, detail=GENERIC_ERROR_MSG)

@router.delete("/{document_id}")
async def delete_document(document_id: int, user: dict = Depends(require_auth)):
    if not database.db_client:
        raise HTTPException(503, "Database not initialized")
    try:
        client = database.db_client.client if hasattr(database.db_client, "client") else database.db_client
        client.table("documents").delete().eq("id", document_id).execute()
        return {"message": f"Document {document_id} deleted successfully"}
    except Exception as e:
        logging.error(f"Document delete error: {e}", exc_info=LOG_EXC_INFO)
        raise HTTPException(500, detail=GENERIC_ERROR_MSG)