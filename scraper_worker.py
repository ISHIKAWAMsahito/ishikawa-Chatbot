import os
import asyncio
import logging
import httpx
from bs4 import BeautifulSoup
from google.generativeai import embed_content # このファイルのimport
from core.database import SupabaseClientManager
from services.document_processor import SimpleDocumentProcessor

# --- 1. 必要な情報を環境変数から読み込む ---
# (RenderのCron Job設定で、Web Serviceと同じ環境変数を設定する必要がある)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_KEY")
EMBEDDING_MODEL = "models/gemini-embedding-001"
COLLECTION_NAME = "student-knowledge-base" # 固定値

logging.basicConfig(level=logging.INFO)

# --- 2. データベースとプロセッサを初期化 ---
try:
    db_client = SupabaseClientManager(url=SUPABASE_URL, key=SUPABASE_SERVICE_KEY)
    simple_processor = SimpleDocumentProcessor(chunk_size=1000, chunk_overlap=200)
    logging.info("DBクライアントとプロセッサの初期化完了。")
except Exception as e:
    logging.error(f"初期化に失敗: {e}")
    exit(1) # エラーで終了

# --- 3. 自動更新したいURLのリストを定義 ---
# (例: これら3つのページを毎日チェックする)
URLS_TO_SCRAPE = [
    "https://www.hokkaido-univcoop.jp/cafeteria-menu/fair.html", # 学食メニュー
    "https://www.hokkaido-univcoop.jp/sgu/",#生協HP
    "https://www.sgu.ac.jp/"# 札幌学院大学HP
]

async def scrape_and_update(url: str):
    """
    documents.py の /scrape ロジックを再利用した関数
    """
    logging.info(f"スクレイプ開始: {url}")
    try:
        # 1. Webサイトからコンテンツを取得
        async with httpx.AsyncClient() as client:
            response = await client.get(url, follow_redirects=True, timeout=10.0)
            response.raise_for_status()

        # 2. HTMLからテキストを抽出
        soup = BeautifulSoup(response.text, 'html.parser')
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        
        target_element = soup.body if soup.body else soup
        body_text = target_element.get_text(separator=' ', strip=True)

        if not body_text:
            logging.warning(f"テキスト抽出失敗 (空): {url}")
            return

        # 3. チャンキング
        filename_from_url = url.split('/')[-1] or url
        docs_to_embed = simple_processor.process_and_chunk(
            filename=f"scrape_{filename_from_url}.txt", 
            content=body_text.encode('utf-8'),
            category="WebScrape-Auto", 
            collection_name=COLLECTION_NAME
        )
        if not docs_to_embed:
            logging.warning(f"チャンク生成失敗: {url}")
            return

        # 4. TODO: ここで「古いデータ」を削除する (重要)
        # 実際には、挿入する前に `db_client.delete_documents_by_source(f"scrape_{filename_from_url}.txt")` 
        # のような関数を実装して、古い情報を削除する必要があります。

        # 5. ベクトル化 & DB挿入
        count = 0
        for doc in docs_to_embed:
            embedding_response = embed_content(model=EMBEDDING_MODEL, content=doc.page_content)
            embedding = embedding_response["embedding"]
            db_client.insert_document(doc.page_content, embedding, doc.metadata)
            count += 1
            await asyncio.sleep(1) # APIレート制限対策

        logging.info(f"スクレイプ成功: {url} ({count} チャンク挿入)")

    except Exception as e:
        logging.error(f"スクレイプ失敗: {url} - {e}")

async def main():
    tasks = [scrape_and_update(url) for url in URLS_TO_SCRAPE]
    await asyncio.gather(*tasks)
    logging.info("自動スクレイピング タスク完了。")

if __name__ == "__main__":
    asyncio.run(main())