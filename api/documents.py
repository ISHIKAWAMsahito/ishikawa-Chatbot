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

# documents.py の process_batch_insert 関数をこれに置き換えてください

async def process_batch_insert(batch_docs: List[Any], embedding_model: str, collection_name: str):
    """
    バッチ処理用のヘルパー関数（リトライ強化版）。
    API制限にかかった場合、60秒待機して再試行します。
    """
    if not batch_docs:
        return 0

    batch_texts = [doc.page_content for doc in batch_docs]
    
    # 最大リトライ回数
    max_retries = 3
    
    for attempt in range(max_retries + 1):
        try:
            # API呼び出し
            embedding_response = genai.embed_content(
                model=embedding_model,
                content=batch_texts,
                task_type="retrieval_document"
            )
            embeddings = embedding_response["embedding"]
            
            # DBへの挿入
            inserted_count = 0
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
            is_quota_error = "429" in str(e) or "quota" in str(e).lower()
            
            if is_quota_error and attempt < max_retries:
                wait_time = 60  # 無料枠の回復を考慮して余裕を持って60秒待機
                logging.warning(f"API制限(429)を検知。{wait_time}秒待機してリトライします... ({attempt + 1}/{max_retries}回目)")
                await asyncio.sleep(wait_time)
                continue # 次のループ（再試行）へ
            else:
                # リトライ回数切れ、または別のエラーの場合は例外を投げる
                logging.error(f"バッチベクトル化エラー（リトライ断念）: {e}")
                raise e


# 12/26 既存の cleaning_instruction を upload でも使えるように関数の外、または共通化して定義。upload と scrape の両方で、処理開始直後に 「同じファイル名（source）のデータがあれば削除する」 処理
COMMON_CLEANING_INSTRUCTION = """
【整形ルール】
1. テキストは読みやすいMarkdown形式に整形してください。
2. もしカレンダーや行事予定表の場合は、「YYYY年M月D日(曜): 内容」の形式に統一してください。
3. 無意味な記号や、単なる装飾（ヘッダー・フッターの繰り返しなど）は削除してください。
"""

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...), 
    category: str = Form("その他"), 
    embedding_model: str = Form("models/gemini-embedding-001"), 
    user: dict = Depends(require_auth)
):
    """
    ファイルを受け取り、Geminiで内容を整形した上で、
    メモリ効率よくバッチ処理でチャンキング・ベクトル化してDBに挿入
    """
    if not database.db_client or not settings.settings_manager or not document_processor.simple_processor:
        raise HTTPException(503, "システムが初期化されていません")
    
    try:
        filename = file.filename
        content = await file.read()
        mime_type = file.content_type
        
        logging.info(f"ファイルアップロード受信: {filename} (MIME: {mime_type})")

        # -------------------------------------------------------
        # 1. 重複防止: 古いデータを完全に削除
        # -------------------------------------------------------
        # ファイル名をユニークIDとして扱い、同名ファイルがあれば削除してから登録します
        logging.info(f"重複チェック: 古いデータ (source: {filename}) を削除中...")
        try:
            database.db_client.client.table("documents").delete().eq("metadata->>source", filename).execute()
        except Exception as e:
            logging.warning(f"削除時の軽微なエラー (初回登録時は無視可): {e}")

        # -------------------------------------------------------
        # 2. Geminiによる読み取り & 整形 (ここが追加・変更点)
        # -------------------------------------------------------
        logging.info("Geminiによるファイル解析と整形を開始します...")
        
        # 解析用モデル (高速なFlashモデル推奨)
        extract_model = genai.GenerativeModel("gemini-2.5-flash")
        
        # ファイルタイプに応じたプロンプト構成
        prompt = f"""
        以下のファイルのテキストを読み取り、整形して出力してください。
        
        {COMMON_CLEANING_INSTRUCTION}
        """

        cleaned_text = ""
        
        try:
            # PDF または テキスト/CSV などをGeminiに渡す
            if "pdf" in mime_type or "text" in mime_type or "csv" in mime_type:
                ai_response = await extract_model.generate_content_async(
                    [prompt, {"mime_type": mime_type, "data": content}]
                )
                cleaned_text = ai_response.text
            else:
                # 画像などその他の場合もトライ、あるいはエラーハンドリング
                ai_response = await extract_model.generate_content_async(
                    [prompt, {"mime_type": mime_type, "data": content}]
                )
                cleaned_text = ai_response.text
                
        except Exception as e:
            logging.error(f"Gemini解析エラー: {e}")
            # エラー時はフェイルセーフとして、生のテキスト抽出を試みるかエラーを返す
            # ここでは簡易的に生データをutf-8デコードして使う例（PDFだと文字化けするので注意）
            if "text" in mime_type:
                cleaned_text = content.decode("utf-8", errors="ignore")
            else:
                raise HTTPException(status_code=500, detail="AIによるファイル読み取りに失敗しました。")

        logging.info(f"AI整形完了: {len(cleaned_text)} 文字")

        # -------------------------------------------------------
        # 3. チャンク化 & 保存 (整形済みテキストを使用)
        # -------------------------------------------------------
        # ここで、ファイルの中身そのものではなく、AIが作った cleaned_text を渡すのがポイント
        collection_name = settings.settings_manager.settings.get("collection", ACTIVE_COLLECTION_NAME)
        
        # 拡張子を .txt に変更して保存（中身はマークダウンテキストになっているため）
        processed_filename = filename
        if not processed_filename.endswith(".txt"):
            processed_filename += ".txt"

        doc_generator = document_processor.simple_processor.process_and_chunk(
            filename=processed_filename, # 内部的にはtxtとして扱う
            content=cleaned_text.encode('utf-8'), # AIのテキストをエンコードして渡す
            category=category, 
            collection_name=collection_name
        )
        
        if doc_generator is None:
             raise HTTPException(status_code=400, detail="ファイル処理の初期化に失敗しました")

        # 以下、バッチ処理は既存コードと同じ
        batch_docs = []
        batch_size = 50
        total_count = 0
        
        for doc in doc_generator:
            # 元のファイル名情報をメタデータに残したい場合はここで上書き
            doc.metadata["source"] = filename 
            
            batch_docs.append(doc)
            if len(batch_docs) >= batch_size:
                inserted = await process_batch_insert(batch_docs, embedding_model, collection_name)
                total_count += inserted
                batch_docs = []
                await asyncio.sleep(0.5)

        if batch_docs:
            inserted = await process_batch_insert(batch_docs, embedding_model, collection_name)
            total_count += inserted

        return {"chunks": total_count, "filename": filename, "message": "AI整形・保存完了"}

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"アップロード処理エラー: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# api/documents.py
 #12/24 Geminiに送るプロンプト（指示書）に、以下の**「カレンダー整形ルール」**を追加しました。丸数字や記号の除去: ① や < > を削除させる。日付の補完: 文脈から「2025年」などを読み取り、YYYY年M月D日 形式に統一させる。重複の削除: 1 (金): 1 金 のような重複を整理させる。
 #12/24 Geminiに送るプロンプト（指示書）に、以下の**「カレンダー整形ルール」**を追加しました。丸数字や記号の除去: ① や < > を削除させる。日付の補完: 文脈から「2025年」などを読み取り、YYYY年M月D日 形式に統一させる。重複の削除: 1 (金): 1 金 のような重複を整理させる。
@router.post("/scrape")
async def scrape_website(
    request: ScrapeRequest, 
    user: dict = Depends(require_auth)
):
    """
    URLからHTML/PDFを取得し、行事予定であれば整形ルールに基づいてキレイなデータにして保存します。
    """
    if not database.db_client or not settings.settings_manager or not document_processor.simple_processor:
        raise HTTPException(503, "システムが初期化されていません")

    logging.info(f"AI Scrapeリクエスト受信: {request.url}")

    try:
        # 1. 接続処理 (大学サイト対策ヘッダー付き)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.google.com/",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
            "Upgrade-Insecure-Requests": "1"
        }

        async with httpx.AsyncClient(verify=False, headers=headers, follow_redirects=True) as client:
            try:
                response = await client.get(request.url, timeout=30.0)
                response.raise_for_status()
                content_body = response.content
            except httpx.RequestError as e:
                logging.error(f"接続エラー: {e}")
                raise HTTPException(status_code=400, detail=f"URLに接続できませんでした: {e}")
            except httpx.HTTPStatusError as e:
                logging.error(f"HTTPエラー: {e.response.status_code}")
                raise HTTPException(status_code=400, detail=f"アクセス拒否: {e.response.status_code}")

        # 2. PDF/HTML判定
        is_pdf_signature = content_body.startswith(b'%PDF')
        content_type_header = response.headers.get("content-type", "").lower()
        
        target_type = "HTML"
        if is_pdf_signature:
            target_type = "PDF"
        elif "application/pdf" in content_type_header:
            if b'<html' in content_body[:1000].lower():
                logging.warning("PDFヘッダーですが中身はHTMLです")
                target_type = "HTML"
            else:
                target_type = "PDF"

        # ---------------------------------------------------------
        # 3. 整形ロジックを組み込んだプロンプトの作成
        # ---------------------------------------------------------
        # ここに先ほどのロジックを「自然言語の指示」として組み込みます
        cleaning_instruction = """
        【重要：出力フォーマットの指定】
        入力データが「行事予定表」や「カレンダー」である場合は、以下のルールで整形してください。
        
        1. **フォーマット統一**: 各行を必ず「YYYY年M月D日(曜): イベント内容」の形式にする。
           - 年や月がヘッダーにある場合は、それを各行の日付に適用して補完すること。
           - 例: "2025年8月1日(金): 定期試験"
        
        2. **ノイズ除去**:
           - 丸数字（①, ⑬など）はすべて削除する。
           - 記号（< > など）は削除する。
           - 日付の重複表記（例: "1 (金): 1 金"）は整理して1つにする。
           - "定期試験" や "夏期休業" などの重要なキーワードは残す。

        3. **対象外データ**: 
           - 予定が入っていない空白の日や、単なる罫線データは出力しない。
           - カレンダー以外の一般的な文章の場合は、読みやすいMarkdown形式で出力する。
        """

        extracted_text = ""
        # バージョン指定付きモデル名を使用
        extract_model = genai.GenerativeModel("gemini-2.5-flash")

        if target_type == "PDF":
            logging.info("PDF解析モード: 整形ルール適用中...")
            prompt = f"""
            このPDFファイルの内容を読み取り、テキストを抽出してください。
            {cleaning_instruction}
            """
            try:
                ai_response = await extract_model.generate_content_async(
                    [prompt, {"mime_type": "application/pdf", "data": content_body}]
                )
                extracted_text = ai_response.text
            except Exception as e:
                logging.error(f"Gemini解析エラー(PDF): {e}")
                raise HTTPException(status_code=500, detail=f"PDF解析失敗: {e}")

        else: # HTML
            logging.info("HTML解析モード: 整形ルール適用中...")
            soup = BeautifulSoup(response.text, 'html.parser')
            for element in soup(["script", "style", "noscript", "iframe", "svg", "header", "footer"]):
                element.decompose()
            target_html = str(soup.body) if soup.body else str(soup)
            
            prompt = f"""
            以下のHTMLからメインコンテンツを抽出してください。
            メニューや広告は除外してください。
            {cleaning_instruction}
            """
            try:
                ai_response = await extract_model.generate_content_async([prompt, target_html])
                extracted_text = ai_response.text
            except Exception as e:
                logging.warning(f"AI解析失敗、フォールバック: {e}")
                extracted_text = soup.get_text(separator=' ', strip=True)

        if not extracted_text:
            raise HTTPException(status_code=400, detail="テキストを抽出できませんでした。")

        # ---------------------------------------------------------
        # 4. 保存処理 (.txtとして保存)
        # ---------------------------------------------------------
        filename_from_url = request.url.split('/')[-1].split('?')[0] or "downloaded_file"
        
        # 拡張子は必ず .txt にする (中身は整形済みテキストなので)
        if filename_from_url.lower().endswith(".pdf"):
            filename_for_processor = filename_from_url[:-4] + ".txt"
        elif not filename_from_url.lower().endswith(".txt"):
            filename_for_processor = filename_from_url + ".txt"
        else:
            filename_for_processor = filename_from_url

        source_name = f"scrape_{filename_from_url}"
        internal_filename = f"scrape_{filename_for_processor}"

        logging.info(f"保存開始: {source_name} (プレビュー: {extracted_text[:50].replace(chr(10), ' ')}...)")

        # 旧データ削除
        try:
            database.db_client.client.table("documents").delete().eq("metadata->>source", source_name).execute()
        except Exception:
            pass

        # チャンク化と保存
        doc_generator = document_processor.simple_processor.process_and_chunk(
            filename=internal_filename, 
            content=extracted_text.encode('utf-8'), 
            category=f"WebScrape({target_type})", 
            collection_name=request.collection_name
        )
        
        # バッチ挿入処理
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
            "message": "処理完了（整形済み）", 
            "type": target_type,
            "preview": extracted_text[:200]  # レスポンスで冒頭を確認できるようにする
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"システムエラー: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"エラーが発生しました: {str(e)}")


@router.get("/collections/{collection_name}/documents")
async def get_documents(collection_name: str):
    if not database.db_client:
        raise HTTPException(503, "DB not initialized")
    return {
        "documents": database.db_client.get_documents_by_collection(collection_name),
        "count": database.db_client.count_chunks_in_collection(collection_name)
    }