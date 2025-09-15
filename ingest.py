import os
import chromadb
import pypdf
import docx
import requests
import re
import uuid
import argparse
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------
# 1. 初期設定とモデルの読み込み
# --------------------------------------------------------------------------
DB_PATH = "./chroma_db"
print("Loading sentence-transformer model...")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
print("Model loaded successfully.")
client = chromadb.PersistentClient(path=DB_PATH)

# --------------------------------------------------------------------------
# 2. テキスト抽出・加工のためのヘルパー関数
# --------------------------------------------------------------------------
def read_pdf_content(file_path: str) -> str:
    """PDFファイルからテキストを抽出する"""
    try:
        reader = pypdf.PdfReader(file_path)
        text = "".join(page.extract_text() for page in reader.pages)
        return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return ""

def read_docx_content(file_path: str) -> str:
    """Wordファイル(.docx)からテキストを抽出する"""
    try:
        doc = docx.Document(file_path)
        text = "\n".join(para.text for para in doc.paragraphs)
        return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
        return ""

def read_txt_content(file_path: str) -> str:
    """テキストファイル(.txt)からテキストを抽出する"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        print(f"Error reading TXT {file_path}: {e}")
        return ""

def scrape_website_text(url: str) -> str:
    """URLからWebサイトのテキストを抽出する"""
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        return re.sub(r'\s+', ' ', text).strip()
    except requests.RequestError as e:
        print(f"Could not fetch URL {url}: {e}")
        return ""
    except Exception as e:
        print(f"Error processing website content for {url}: {e}")
        return ""

def split_text_into_chunks(text: str, chunk_size: int = 800, chunk_overlap: int = 100):
    """テキストを指定したサイズに分割する"""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

# --------------------------------------------------------------------------
# 3. メイン処理
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Upload documents to ChromaDB using a local embedding model.")
    parser.add_argument("--collection", type=str, required=True, help="Name of the ChromaDB collection.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to a single local file (PDF, DOCX, TXT).")
    group.add_argument("--url", type=str, help="URL of a website to scrape.")
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # 追加点：フォルダをまとめて処理するための --dir オプションを追加
    # ----------------------------------------------------------------
    group.add_argument("--dir", type=str, help="Path to a directory containing files to upload.")
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

    args = parser.parse_args()
    collection_name = args.collection

    # --- フォルダ一括処理 ---
    if args.dir:
        if not os.path.isdir(args.dir):
            print(f"Error: Directory not found at {args.dir}")
            return
        
        collection = client.get_or_create_collection(name=collection_name)
        total_chunks_added = 0
        files_processed = 0

        for filename in sorted(os.listdir(args.dir)):
            file_path = os.path.join(args.dir, filename)
            if not os.path.isfile(file_path):
                continue
            
            source_text = ""
            if filename.lower().endswith(".pdf"):
                source_text = read_pdf_content(file_path)
            elif filename.lower().endswith(".docx"):
                source_text = read_docx_content(file_path)
            elif filename.lower().endswith(".txt"):
                source_text = read_txt_content(file_path)
            else:
                continue # 対応外のファイルはスキップ

            if not source_text:
                print(f"Could not extract text from {filename}, skipping.")
                continue

            print(f"\n--- Processing file: {filename} ---")
            chunks = split_text_into_chunks(source_text)
            if not chunks:
                print(f"Text in {filename} is too short, skipping.")
                continue
            
            print(f"Split into {len(chunks)} chunks. Embedding...")
            embeddings = model.encode(chunks, show_progress_bar=True).tolist()
            
            collection.add(
                embeddings=embeddings, documents=chunks,
                ids=[str(uuid.uuid4()) for _ in chunks],
                metadatas=[{"filename": filename} for _ in chunks]
            )
            total_chunks_added += len(chunks)
            files_processed += 1
        
        print("\n" + "="*50)
        print("✅ Successfully completed directory processing!")
        print(f"   Collection: {collection_name}")
        print(f"   Files processed: {files_processed}")
        print(f"   Total chunks added: {total_chunks_added}")
        print(f"   Total documents in collection now: {collection.count()}")
        print(f"   Database saved in: '{DB_PATH}' folder")
        print("="*50)
        return

    # --- 単体ファイルまたはURLの処理 (従来通り) ---
    source_text = ""
    metadata = {}
    if args.file:
        # (単体ファイル処理のロジックは変更なし)
        print(f"Processing file: {args.file}")
        if not os.path.exists(args.file):
            print(f"Error: File not found at {args.file}")
            return
        if args.file.lower().endswith(".pdf"): source_text = read_pdf_content(args.file)
        elif args.file.lower().endswith(".docx"): source_text = read_docx_content(args.file)
        elif args.file.lower().endswith(".txt"): source_text = read_txt_content(args.file)
        else:
            print(f"Error: Unsupported file type. Please use .pdf, .docx, or .txt.")
            return
        metadata = {"filename": os.path.basename(args.file)}

    elif args.url:
        # (URL処理のロジックは変更なし)
        print(f"Processing URL: {args.url}")
        source_text = scrape_website_text(args.url)
        metadata = {"source_url": args.url}

    if not source_text:
        print("No text could be extracted. Aborting.")
        return

    chunks = split_text_into_chunks(source_text)
    if not chunks:
        print("Text was too short to be chunked. Aborting.")
        return
    
    collection = client.get_or_create_collection(name=collection_name)
    print(f"Split into {len(chunks)} chunks. Embedding...")
    embeddings = model.encode(chunks, show_progress_bar=True).tolist()
    collection.add(
        embeddings=embeddings, documents=chunks,
        ids=[str(uuid.uuid4()) for _ in chunks],
        metadatas=[metadata for _ in chunks]
    )
    print("\n" + "="*50)
    print("✅ Successfully added data to ChromaDB!")
    print(f"   Collection: {collection.name}")
    print(f"   Chunks added: {len(chunks)}")
    print(f"   Total documents in collection: {collection.count()}")
    print(f"   Database saved in: '{DB_PATH}' folder")
    print("="*50)

if __name__ == "__main__":
    main()

