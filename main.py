<<<<<<< HEAD
# --- Standard Library Imports ---
import os
import uvicorn
import logging
import json
from typing import List
import uuid
import io

# --- Third-party Library Imports ---
# [重要] 以下のライブラリが requirements.txt に含まれていることを確認してください
# fastapi, uvicorn, python-dotenv, google-generativeai, supabase, requests, beautifulsoup4, PyPDF2, python-docx
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
from supabase import create_client, Client
import requests
from bs4 import BeautifulSoup
import PyPDF2
from docx import Document as DocxDocument

# --- 初期設定 ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
load_dotenv()

# --- 環境変数と定数の設定 ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not GEMINI_API_KEY:
    raise ValueError("環境変数「GEMINI_API_KEY」が設定されていません。")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("環境変数「SUPABASE_URL」と「SUPABASE_KEY」が設定されていません。")

genai.configure(api_key=GEMINI_API_KEY)
ACTIVE_COLLECTION_NAME = "student-knowledge-base"

# ==============================================================================
# ▼▼▼ 内部コンポーネント (以前は別ファイルだったもの) ▼▼▼
# ==============================================================================

def split_text(text: str, max_length: int = 1000, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_length
        chunks.append(text[start:end])
        start += max_length - overlap
    return chunks

class DocumentProcessor:
    def process(self, filename: str, content: bytes) -> List[str]:
        text = ""
        try:
            if filename.lower().endswith(".pdf"):
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
            elif filename.lower().endswith(".docx"):
                doc = DocxDocument(io.BytesIO(content))
                for para in doc.paragraphs:
                    text += para.text + "\n"
            else:
                text = content.decode('utf-8', errors='ignore')
        except Exception as e:
            logging.error(f"ファイル処理エラー ({filename}): {e}")
            return []
        return split_text(text)

class WebScraper:
    def scrape(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            for script_or_style in soup(["script", "style", "header", "footer", "nav", "aside"]):
                script_or_style.decompose()
            return " ".join(t.strip() for t in soup.stripped_strings)
        except Exception as e:
            logging.error(f"スクレイピングエラー ({url}): {e}")
            return ""

class LogManager:
    def generate_log_id(self) -> str:
        return str(uuid.uuid4())
    def save_log(self, log_id, query, response, context, category):
        logging.info(f"LOG [{log_id}]: Q='{query}' A='{response[:80].strip()}...' Cat='{category}'")

class SupabaseClient:
    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)

    def get_db_type(self) -> str:
        return "supabase"

    def insert_document(self, content: str, embedding: List[float], metadata: dict):
        self.client.table("documents").insert({
            "content": content, "embedding": embedding, "metadata": metadata
        }).execute()

    def search_documents(self, collection_name: str, category: str, embedding: List[float], match_count: int) -> List[dict]:
        params = {
            "p_collection_name": collection_name,
            "p_category": category,
            "p_query_embedding": embedding,
            "p_match_count": match_count
        }
        return self.client.rpc("match_documents", params).execute().data

    def get_documents_by_collection(self, collection_name: str) -> List[dict]:
        return self.client.table("documents").select("id, metadata").eq("metadata->>collection_name", collection_name).execute().data
    
    def count_chunks_in_collection(self, collection_name: str) -> int:
        return self.client.table("documents").select("id", count='exact').eq("metadata->>collection_name", collection_name).execute().count

    def get_distinct_categories(self, collection_name: str) -> List[str]:
        try:
            # [重要] この機能には、Supabaseに 'get_distinct_categories' というSQL関数が必要です
            result = self.client.rpc("get_distinct_categories", {"p_collection_name": collection_name}).execute()
            return [item['category'] for item in result.data if item.get('category')]
        except Exception as e:
            logging.error(f"RPC 'get_distinct_categories' の呼び出しエラー: {e}")
            return ["その他"]

db_instance = None
def get_db_client() -> SupabaseClient:
    global db_instance
    if db_instance is None:
        db_instance = SupabaseClient(url=SUPABASE_URL, key=SUPABASE_KEY)
    return db_instance

# ==============================================================================

# --- FastAPIアプリケーションのインスタンス化 ---
app = FastAPI()
db_client = get_db_client()
log_manager = LogManager()
document_processor = DocumentProcessor()
web_scraper = WebScraper()

# --- データモデル定義 ---
class ChatQuery(BaseModel):
    query: str; model: str; collection: str; embedding_model: str; top_k: int

class ScrapeRequest(BaseModel):
    url: str; collection_name: str; category: str; embedding_model: str

# --- APIエンドポイント ---
@app.get("/", response_class=HTMLResponse)
@app.get("/admin", response_class=HTMLResponse)
async def serve_admin_page():
    file_path = os.path.join(os.path.dirname(__file__), "admin.html")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    raise HTTPException(status_code=404, detail="Admin page not found")

=======
import os
import re
import uvicorn
import logging
import json
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, WebSocket
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel # [修正点] pantic -> pydantic
from typing import List, Optional
import google.generativeai as genai
from dotenv import load_dotenv

# 独自のモジュール (データベース接続や文書処理など)
from custom_components.db_client import SupabaseClient, get_db_client
from custom_components.document_processor import DocumentProcessor, split_text
from custom_components.web_scraper import WebScraper
from custom_components.log_manager import LogManager

# --- 初期設定 ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
load_dotenv()

# --- 環境変数と定数の設定 ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("環境変数「GEMINI_API_KEY」が設定されていません。")

# Google Generative AIの設定
genai.configure(api_key=GEMINI_API_KEY)

# 固定コレクション名 (UIの変更に合わせる)
ACTIVE_COLLECTION_NAME = "student-knowledge-base"

# --- FastAPIアプリケーションのインスタンス化 ---
app = FastAPI()
db_client: SupabaseClient = get_db_client()
log_manager = LogManager()
document_processor = DocumentProcessor()
web_scraper = WebScraper()

# --- WebSocket管理 ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

websocket_manager = ConnectionManager()
current_settings = {
    "model": "gemini-1.5-flash-latest",
    "collection": ACTIVE_COLLECTION_NAME,
    "embedding_model": "text-embedding-004",
    "top_k": 5
}

# --- データモデル定義 (Pydantic) ---
class ChatQuery(BaseModel):
    query: str
    model: str
    collection: str
    embedding_model: str
    top_k: int

class ScrapeRequest(BaseModel):
    url: str
    collection_name: str
    category: str
    embedding_model: str

# --- APIエンドポイント ---
@app.get("/", response_class=HTMLResponse)
@app.get("/admin", response_class=HTMLResponse)
async def serve_admin_page(request: Request):
    file_path = os.path.join(os.path.dirname(__file__), "admin.html")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    raise HTTPException(status_code=404, detail="Admin page not found")


>>>>>>> 60d72e17b4e23974c67d915281db7df0c640ad59
@app.get("/health")
async def health_check():
    return {"status": "ok", "database": db_client.get_db_type()}

@app.get("/gemini/status")
async def gemini_status():
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        return {"connected": True, "models": models}
    except Exception as e:
<<<<<<< HEAD
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections/{collection_name}/documents")
async def get_documents(collection_name: str):
    return {
        "documents": db_client.get_documents_by_collection(collection_name),
        "count": db_client.count_chunks_in_collection(collection_name)
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), embedding_model: str = Form(...)):
    try:
        category = file.filename.split('_')[0] if '_' in file.filename else "未分類"
        content = await file.read()
        chunks = document_processor.process(file.filename, content)
        for chunk in chunks:
            metadata = {"source": file.filename, "collection_name": ACTIVE_COLLECTION_NAME, "category": category}
            embedding = genai.embed_content(model=embedding_model, content=chunk)["embedding"]
            db_client.insert_document(chunk, embedding, metadata)
        return {"chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scrape")
async def scrape_website(req: ScrapeRequest):
    try:
        content = web_scraper.scrape(req.url)
        if not content: raise HTTPException(status_code=400, detail="コンテンツを取得できませんでした。")
        chunks = split_text(content)
        for chunk in chunks:
            metadata = {"source": req.url, "collection_name": req.collection_name, "category": req.category}
            embedding = genai.embed_content(model=req.embedding_model, content=chunk)["embedding"]
            db_client.insert_document(chunk, embedding, metadata)
        return {"chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def chat_streamer(query_data: ChatQuery):
    log_id = log_manager.generate_log_id()
    yield f"data: {json.dumps({'log_id': log_id})}\n\n"
    try:
        available_categories = db_client.get_distinct_categories(query_data.collection)
        if not available_categories:
            yield f"data: {json.dumps({'content': '現在参照できる知識がありません。'})}\n\n"
            return
        
        classification_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"利用可能なカテゴリ: {', '.join(available_categories)}\nユーザーの質問: 「{query_data.query}」\n質問に最も関連するカテゴリ名を一つだけ出力してください。該当がなければ「その他」と出力。"
        response = await classification_model.generate_content_async(prompt)
        classified_category = response.text.strip()
        
        message_content = f'（カテゴリ: {classified_category} を検索中...）\n'
        yield f"data: {json.dumps({'content': message_content})}\n\n"

        query_embedding = genai.embed_content(model=query_data.embedding_model, content=query_data.query)["embedding"]
        search_results = db_client.search_documents(
            collection_name=query_data.collection, category=classified_category,
            embedding=query_embedding, match_count=query_data.top_k
=======
        return HTTPException(status_code=500, detail=str(e))

@app.get("/collections/{collection_name}/documents")
async def get_documents_in_collection(collection_name: str):
    documents = db_client.get_documents_by_collection(collection_name)
    count = db_client.count_chunks_in_collection(collection_name)
    return {"documents": documents, "count": count}

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection_name: str = Form(ACTIVE_COLLECTION_NAME),
    embedding_model: str = Form("text-embedding-004")
):
    try:
        category = file.filename.split('_')[0] if '_' in file.filename else "未分類"
        
        content = await file.read()
        chunks = document_processor.process(file.filename, content)
        
        logging.info(f"ファイル '{file.filename}' を {len(chunks)} チャンクに分割しました。カテゴリ: {category}")

        for chunk in chunks:
            metadata = {"source": file.filename, "collection_name": collection_name, "category": category}
            embedding = genai.embed_content(model=embedding_model, content=chunk)["embedding"]
            db_client.insert_document(chunk, embedding, metadata)
            
        return {"chunks": len(chunks)}
    except Exception as e:
        logging.error(f"アップロード処理中にエラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scrape")
async def scrape_website(req: ScrapeRequest):
    try:
        content = web_scraper.scrape(req.url)
        if not content:
            raise HTTPException(status_code=400, detail="ウェブサイトからコンテンツを取得できませんでした。")
        
        chunks = split_text(content)
        logging.info(f"URL '{req.url}' を {len(chunks)} チャンクに分割しました。カテゴリ: {req.category}")

        for chunk in chunks:
            metadata = {"source": req.url, "collection_name": req.collection_name, "category": req.category}
            embedding = genai.embed_content(model=req.embedding_model, content=chunk)["embedding"]
            db_client.insert_document(chunk, embedding, metadata)
            
        return {"chunks": len(chunks)}
    except Exception as e:
        logging.error(f"スクレイピング処理中にエラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def chat_streamer(query_data: ChatQuery):
    log_id = log_manager.generate_log_id()
    yield f"data: {json.dumps({'log_id': log_id})}\n\n"

    try:
        available_categories = db_client.get_distinct_categories(query_data.collection)
        if not available_categories:
            yield "data: 申し訳ありませんが、現在参照できる知識がありません。\n\n"
            return

        classification_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"""ユーザーの質問内容に最も関連するカテゴリを、以下のリストから一つだけ選んでください。
        利用可能なカテゴリ: {', '.join(available_categories)}
        ユーザーの質問: 「{query_data.query}」
        最も関連性の高いカテゴリ名を一つだけ出力してください。該当するものがない場合は「その他」と出力してください。"""
        
        response = await classification_model.generate_content_async(prompt)
        classified_category = response.text.strip()
        logging.info(f"質問をカテゴリ '{classified_category}' に分類しました。")

        message_content = f'（カテゴリ: {classified_category} を検索中...）\n'
        json_payload = json.dumps({'content': message_content})
        yield f"data: {json_payload}\n\n"

        query_embedding = genai.embed_content(model=query_data.embedding_model, content=query_data.query)["embedding"]
        
        search_results = db_client.search_documents(
            collection_name=query_data.collection,
            category=classified_category,
            embedding=query_embedding,
            match_count=query_data.top_k
>>>>>>> 60d72e17b4e23974c67d915281db7df0c640ad59
        )
        context = "\n".join([doc['content'] for doc in search_results]) or "関連情報は見つかりませんでした。"

<<<<<<< HEAD
        final_model = genai.GenerativeModel(query_data.model)
        final_prompt = f"参考情報:\n---\n{context}\n---\n質問: 「{query_data.query}」\n\n参考情報に基づいて、学生からの質問に親切かつ丁寧に回答してください。"
        stream = await final_model.generate_content_async(final_prompt, stream=True)
        
        full_response_text = ""
        async for chunk in stream:
            full_response_text += chunk.text
            yield f"data: {json.dumps({'content': chunk.text})}\n\n"
        
        log_manager.save_log(log_id, query_data.query, full_response_text, context, classified_category)
    except Exception as e:
        logging.error(f"チャット処理エラー: {e}")
        yield f"data: {json.dumps({'content': f'エラーが発生しました: {e}'})}\n\n"

@app.post("/chat")
async def chat_endpoint(query: ChatQuery):
    return StreamingResponse(chat_streamer(query), media_type="text/event-stream")
=======
        context = "\n".join([doc['content'] for doc in search_results])
        if not context:
            context = "関連情報は見つかりませんでした。"

        final_model = genai.GenerativeModel(query_data.model)
        final_prompt = f"""以下の参考情報に基づいて、学生からの質問に親切かつ丁寧に回答してください。

        参考情報:
        ---
        {context}
        ---
        質問:
        「{query_data.query}」

        回答:
        """
        
        stream = await final_model.generate_content_async(final_prompt, stream=True)
        full_response_text = ""
        async for chunk in stream:
            full_response_text += chunk.text
            yield f"data: {json.dumps({'content': chunk.text})}\n\n"
        
        log_manager.save_log(log_id, query_data.query, full_response_text, context, classified_category)

    except Exception as e:
        logging.error(f"チャット処理中にエラー: {e}")
        yield f"data: {json.dumps({'content': f'エラーが発生しました: {e}'})}\n\n"


@app.post("/chat")
async def chat_endpoint(query: ChatQuery):
    return StreamingResponse(chat_streamer(query), media_type="text/event-stream")

# --- 残りのエンドポイント ---
# ... (変更なし)
>>>>>>> 60d72e17b4e23974c67d915281db7df0c640ad59

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

<<<<<<< HEAD
=======
# [修正点] ファイルの末尾にあった不要なGitコマンドを削除

>>>>>>> 60d72e17b4e23974c67d915281db7df0c640ad59
