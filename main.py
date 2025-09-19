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


@app.get("/health")
async def health_check():
    return {"status": "ok", "database": db_client.get_db_type()}

@app.get("/gemini/status")
async def gemini_status():
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        return {"connected": True, "models": models}
    except Exception as e:
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
        )

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# [修正点] ファイルの末尾にあった不要なGitコマンドを削除

