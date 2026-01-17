import os
import logging
import asyncio
import glob
from typing import List
import PIL.Image

import google.generativeai as genai
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
import re

# --------------------------------------------------------------------------
# ★ 設定: 環境変数の読み込み
# --------------------------------------------------------------------------
ENV_PATH = r"C:\dev\ishikawa-Chatbot.env"
load_dotenv(dotenv_path=ENV_PATH)

# 環境変数の取得
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY or not GEMINI_API_KEY:
    raise ValueError("環境変数が設定されていません。")

genai.configure(api_key=GEMINI_API_KEY)

# DB設定
ACTIVE_COLLECTION_NAME = "student-knowledge-base"

# ★モデル設定
# 埋め込みモデルは変更せず維持
EMBEDDING_MODEL = "models/gemini-embedding-001"

# OCRモデル: 指摘に基づき gemini-2.5-flash を採用
# 2.5 Flashは1.5に比べてレイテンシとコストパフォーマンスが向上しています
VISION_MODEL = "gemini-2.5-flash"

# ログ設定
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING) 

# --------------------------------------------------------------------------
# クラス・関数定義
# --------------------------------------------------------------------------

class SupabaseClientManager:
    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)

    def insert_document(self, content: str, embedding: List[float], metadata: dict):
        try:
            self.client.table("documents").insert({
                "content": content,
                "embedding": embedding,
                "metadata": metadata
            }).execute()
        except Exception as e:
            logging.error(f"Supabase挿入エラー: {e}")

db_client = SupabaseClientManager(url=SUPABASE_URL, key=SUPABASE_KEY)

def _clean_text(text: str) -> str:
    if not text: return ""
    text = text.replace("\t", " ").replace("　", " ")
    text = re.sub(r'(\n|\r|\r\n)+', '\n', text) 
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def _convert_to_markdown(text: str) -> str:
    # 既存の正規表現ロジック
    text = re.sub(r'^\s*([ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ]+)．(.*)', r'# \1．\2', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*([０-９]+| [１-９]) ．(.*)', r'## \1 ．\2', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*（\s*([０-９]+|[１-９])\s*）(.*)', r'### （\1）\2', text, flags=re.MULTILINE)
    return text

async def extract_text_from_image(image_path: str) -> str:
    try:
        img = PIL.Image.open(image_path)
        # 指定されたVisionモデルを使用
        model = genai.GenerativeModel(VISION_MODEL)
        
        prompt = """
        あなたはこの画像の情報をデータベース化するプロフェッショナルです。
        以下の厳格なルールに従って、画像内のテキストを書き起こしてください。

        1. **完全性**: 文字を一字一句正確に書き出してください。
        2. **構造化**: 文書の構造に合わせて、大見出し(#)、中見出し(##)、小見出し(###)を適切に使用してください。
        3. **表と注釈の維持**: 
           - 表はMarkdown形式で出力してください。
           - **重要:** 表の下にある注釈（※、①、注意書きなど）は、表と密接に関連する情報です。表の直後に配置し、絶対に表から離さないでください。
        4. **ノイズ除去**: 画像自体の説明（「これは～の画像です」等）は不要です。
        """
        
        response = await model.generate_content_async([prompt, img])
        return response.text
    except Exception as e:
        if "429" in str(e):
            logging.critical("★警告: APIのリクエスト上限に達しました。")
        logging.error(f"画像解析エラー ({os.path.basename(image_path)}): {e}")
        return ""

# --------------------------------------------------------------------------
# メイン処理
# --------------------------------------------------------------------------

async def process_image_directory(directory_path: str, category: str, collection_name: str):
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp']
    raw_files = []
    
    for ext in extensions:
        raw_files.extend(glob.glob(os.path.join(directory_path, ext)))
        raw_files.extend(glob.glob(os.path.join(directory_path, ext.upper())))
    
    image_files = sorted(list(set(raw_files)))

    if not image_files:
        logging.error(f"指定フォルダに画像が見つかりません: {directory_path}")
        return

    logging.info(f"=== 画像処理開始 ===")
    logging.info(f"対象フォルダ: {directory_path}")
    logging.info(f"処理ファイル数: {len(image_files)} 枚") 
    logging.info(f"使用モデル: OCR={VISION_MODEL}, Embedding={EMBEDDING_MODEL}")

    # ヘッダー分割の設定
    headers_to_split_on = [("#", "Header1"), ("##", "Header2"), ("###", "Header3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    
    # チャンク分割の設定
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)

    for i, filepath in enumerate(image_files):
        filename = os.path.basename(filepath)
        logging.info(f"--- [{i+1}/{len(image_files)}] {filename} の解析中... ---")

        # 1. 画像からテキスト抽出
        raw_text = await extract_text_from_image(filepath)
        
        if not raw_text:
            logging.warning(f"  -> テキスト取得失敗/制限。スキップ。")
            continue

        clean_text = _clean_text(raw_text)
        md_text = _convert_to_markdown(clean_text)

        # 2. Markdown構造で親チャンクを作成
        parent_chunks = markdown_splitter.split_text(md_text)
        if not parent_chunks:
            from langchain_core.documents import Document
            parent_chunks = [Document(page_content=md_text, metadata={})]

        # 3. 親チャンクを基に子チャンクを作成＆保存
        child_chunks_list = []
        for parent in parent_chunks:
            # 親自体が小さい場合はそのまま使う
            if len(parent.page_content) < 1200:
                child_texts = [parent.page_content]
            else:
                child_texts = text_splitter.split_text(parent.page_content)

            for child_text in child_texts:
                meta = {
                    **parent.metadata,
                    # ★検索精度向上のための重要修正: 親テキストを保持
                    "parent_content": parent.page_content, 
                    "source": filename,
                    "original_path": filepath,
                    "collection_name": collection_name,
                    "category": category,
                    "element_type": "image_ocr",
                    "model_version": VISION_MODEL
                }
                child_chunks_list.append((child_text, meta))

        # 4. ベクトル化とDB保存
        for chunk_text, metadata in child_chunks_list:
            try:
                embedding_resp = genai.embed_content(model=EMBEDDING_MODEL, content=chunk_text)
                db_client.insert_document(chunk_text, embedding_resp["embedding"], metadata)
                # Gemini 2.5 Flashは高速ですが、念のため短い待機時間を設定
                await asyncio.sleep(1) 
            except Exception as e:
                if "429" in str(e):
                    logging.warning("  Embedding API制限。20秒待機してリトライ...")
                    await asyncio.sleep(20)
                    try:
                        embedding_resp = genai.embed_content(model=EMBEDDING_MODEL, content=chunk_text)
                        db_client.insert_document(chunk_text, embedding_resp["embedding"], metadata)
                    except: pass
                else:
                    logging.error(f"  -> DB保存エラー: {e}")

        # 1枚完了後の待機時間 (2.5 Flashはレート制限が緩いため、35秒→5秒に短縮可能ですが、安全を見て5秒としました)
        logging.info(f"  -> {filename} 完了。待機中(5s)...")
        await asyncio.sleep(5)

    logging.info("=== 全画像の処理が完了しました ===")

async def main():
    # パス設定
    target_dir = r"C:\dev\チャットボットdb\経済経営学部　心理学部　履修要項"
    target_category = "rules" 

    if not os.path.exists(target_dir):
        logging.error(f"エラー: 指定されたフォルダが存在しません。\nパス: {target_dir}")
        return

    logging.info(f"ターゲットディレクトリ: {target_dir}")
    logging.info(f"カテゴリ: {target_category}")
    
    await process_image_directory(target_dir, target_category, ACTIVE_COLLECTION_NAME)

if __name__ == "__main__":
    asyncio.run(main())