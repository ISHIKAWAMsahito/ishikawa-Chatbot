# --------------------------------------------------------------------------
# 1. ライブラリのインポート
# --------------------------------------------------------------------------
import os
import json
import httpx
import chromadb
import traceback
import csv
from datetime import datetime, timezone, timedelta
import urllib3
import pypdf
import docx
from contextlib import asynccontextmanager
import uuid
import io
import re
from typing import Dict, Any, Optional, List
import asyncio
import google.generativeai as genai
import pandas as pd

from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from fastapi import Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse, Response
from pydantic import BaseModel
from bs4 import BeautifulSoup
from prometheus_fastapi_instrumentator import Instrumentator

try:
    from dotenv import load_dotenv
    load_dotenv()
    print(".envファイルをロードしました。")
except ImportError:
    print(".envファイルが見つからないか、python-dotenvがインストールされていません。")
    pass

# --------------------------------------------------------------------------
# 2. 環境変数と基本設定
# --------------------------------------------------------------------------
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY: raise ValueError("環境変数 'GEMINI_API_KEY' が設定されていません。")
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"重大なエラー: Gemini APIキーの設定に失敗しました。 - {e}")
    GEMINI_API_KEY = None

AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY")

print("=" * 50)
print("起動時設定確認:")
print(f"  Auth0 CLIENT_ID: {'設定済み' if AUTH0_CLIENT_ID else '未設定'}")
print(f"  Auth0 DOMAIN: {AUTH0_DOMAIN or '未設定'}")
print(f"  Auth0 CLIENT_SECRET: {'設定済み' if AUTH0_CLIENT_SECRET else '未設定'}")
print(f"  APP_SECRET_KEY: {'設定済み' if APP_SECRET_KEY else '未設定'}")
print("=" * 50)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RENDER_DISK_PATH = "/var/data"
DB_PATH = os.path.join(RENDER_DISK_PATH, "chroma_db") if os.path.exists(RENDER_DISK_PATH) else os.path.join(BASE_DIR, "chroma_db")
FEEDBACK_FILE_PATH = os.path.join(RENDER_DISK_PATH, "feedback.csv") if os.path.exists(RENDER_DISK_PATH) else os.path.join(BASE_DIR, "feedback.csv")
SETTINGS_FILE_PATH = os.path.join(BASE_DIR, "shared_settings.json")
JST = timezone(timedelta(hours=+9), 'JST')

# --------------------------------------------------------------------------
# 3. アプリケーションのライフサイクルと初期化
# --------------------------------------------------------------------------
db_client = None
settings_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_client, settings_manager
    print("--- アプリケーション起動処理開始 ---")
    settings_manager = SettingsManager()
    try:
        if not os.path.exists(FEEDBACK_FILE_PATH):
            with open(FEEDBACK_FILE_PATH, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['log_id', 'timestamp', 'query', 'response', 'rating', 'category', 'has_specific_info'])
        db_client = chromadb.PersistentClient(path=DB_PATH)
        print("ChromaDBクライアントの初期化完了。")
    except Exception as e:
        print(f"致命的エラー: ライフサイクル初期化中にエラーが発生しました - {e}\n{traceback.format_exc()}")
    print("--- 起動処理完了 ---")
    yield
    print("--- アプリケーション終了処理 ---")

app = FastAPI(lifespan=lifespan)

if APP_SECRET_KEY:
    app.add_middleware(SessionMiddleware, secret_key=APP_SECRET_KEY)
else:
    print("警告: APP_SECRET_KEYが設定されていないため、セッション機能（ログイン）は動作しません。")

oauth = OAuth()
if all([AUTH0_CLIENT_ID, AUTH0_CLIENT_SECRET, AUTH0_DOMAIN]):
    oauth.register(
        name='auth0', client_id=AUTH0_CLIENT_ID, client_secret=AUTH0_CLIENT_SECRET,
        server_metadata_url=f'https://{AUTH0_DOMAIN}/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid profile email'},
    )
else:
    print("警告: Auth0の設定が不完全なため、認証機能は動作しません。")

Instrumentator().instrument(app).expose(app)

# --------------------------------------------------------------------------
# 4. ヘルパー関数とPydanticモデル
# --------------------------------------------------------------------------
class SettingsManager:
    def __init__(self):
        self.settings = {"model": "gemini-1.5-flash-latest", "collection": "default", "embedding_model": "text-embedding-004", "top_k": 5}
        self.websocket_connections: List[WebSocket] = []
        self.load_settings()
    def load_settings(self):
        try:
            if os.path.exists(SETTINGS_FILE_PATH):
                with open(SETTINGS_FILE_PATH, 'r', encoding='utf-8') as f: self.settings.update(json.load(f))
        except Exception as e: print(f"設定ファイルの読み込みエラー: {e}")
    def save_settings(self):
        try:
            with open(SETTINGS_FILE_PATH, 'w', encoding='utf-8') as f: json.dump(self.settings, f, ensure_ascii=False, indent=2)
        except Exception as e: print(f"設定ファイルの保存エラー: {e}")
    async def update_settings(self, new_settings: Dict[str, Any]):
        self.settings.update(new_settings)
        self.save_settings()
        await self.broadcast_settings()
    async def add_websocket(self, websocket: WebSocket):
        await websocket.accept()
        self.websocket_connections.append(websocket)
    def remove_websocket(self, websocket: WebSocket):
        self.websocket_connections.remove(websocket)
    async def broadcast_settings(self):
        message = {"type": "settings_update", "data": self.settings}
        disconnected_sockets = [conn for conn in self.websocket_connections if conn.client_state.name != 'CONNECTED']
        for socket in disconnected_sockets: self.websocket_connections.remove(socket)
        await asyncio.gather(*[conn.send_json(message) for conn in self.websocket_connections], return_exceptions=True)

class ClientChatRequest(BaseModel): query: str
class ChatRequest(BaseModel):
    query: str; model: str; collection: str; embedding_model: str; top_k: int
class CreateCollectionRequest(BaseModel): name: str
class SettingsUpdateRequest(BaseModel):
    model: str; collection: str; embedding_model: str; top_k: int
class FeedbackRequest(BaseModel): log_id: str; rating: str
class ScrapeRequest(BaseModel):
    url: str; collection_name: str; embedding_model: str

# --- カテゴリとキーワードのマッピング ---
KEYWORD_MAP = {
    "授業": ["学事暦", "シラバス", "授業計画", "年間スケジュール", "夏休み", "冬休み", "春休み", "夏季休業", "冬季休業", "春季休業"],
    "カリキュラム": ["カリキュラム", "履修",  "科目", "単位", "授業", "必修", "選択"],
    "欠席手続き": ["忌引き", "休む", "欠席", "休講", "病気", "インフルエンザ","欠席届", "公認欠席", "特別欠席"],
    "試験": ["試験", "テスト", "追試", "再試", "成績", "レポート", "課題"],
    "証明書": ["証明書", "学割", "発行", "学生証","学籍","休学","復学","退学","再入学","転学部", "転学科", "卒業", "在学", "卒業見込"],
    "留学": ["留学", "海外", "協定校", "IEC", "国際交流"],
    "学生支援": ["サークル", "部活", "ボランティア", "障がい", "障害"],
    "経済支援": ["奨学金", "授業料", "学費", "免除", "支援金", "学費支援","家計急変", "被災学生", "授業料減免"],
    "資格": ["資格", "免許", "講座", "キャリア"],
    "相談": ["相談", "カウンセリング", "悩み", "メンタル", "ハラスメント", "トラブル"],
    "施設": ["図書館", "食堂", "購買", "場所", "どこ", "Wi-Fi", "PC", "パソコン", "教室", "体育館", "グラウンド", "駐車場"],
    "生協": ["生協", "コープ", "教科書", "共済", "組合員","購買", "食堂", "書籍", "パソコン", "カフェテリア", "学内ショップ"],
}
CATEGORY_INFO = {
    "授業": {"url_to_summarize": "https://www.sgu.ac.jp/information/schedule.html", "static_response": "学事暦や年間スケジュールに関するご質問ですね。\n- **夏期休業期間（夏休み）**: 8月上旬～9月中旬\n- **冬季休業期間（冬休み）**: 12月下旬～1月上旬\n- **春季休業期間（春休み）**: 2月上旬～3月下旬\n正確な日付は、[2025年度 学事暦](https://www.sgu.ac.jp/information/schedule.html)でご確認ください。"},
    "証明書": {"url_to_summarize": "https://www.sgu.ac.jp/campuslife/support/certification.html", "static_response": "各種証明書の発行については、[諸証明・各種願出・届出について](https://www.sgu.ac.jp/campuslife/support/certification.html)のページをご確認ください。"},
    "経済支援": {"url_to_summarize": "https://www.sgu.ac.jp/campuslife/scholarship/", "static_response": "奨学金や授業料減免については、[奨学金制度](https://www.sgu.ac.jp/campuslife/scholarship/)や[授業料減免制度](https://www.sgu.ac.jp/tuition/j09tjo00000f665g.html)のページをご確認ください。"},
    # 他のカテゴリ情報も同様に追加
}

# --- テキスト処理・Webスクレイピング関数 ---
async def read_file_content(file: UploadFile):
    filename = file.filename or ""
    content_bytes = await file.read()
    text = ""
    try:
        if filename.endswith(".pdf"):
            reader = pypdf.PdfReader(io.BytesIO(content_bytes))
            text = "".join(page.extract_text() for page in reader.pages)
        elif filename.endswith(".docx"):
            doc = docx.Document(io.BytesIO(content_bytes))
            text = "\n".join(para.text for para in doc.paragraphs)
        elif filename.endswith(".txt") or filename.endswith(".md"):
            text = content_bytes.decode('utf-8')
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file {filename}: {str(e)}")
    return re.sub(r'\s+', ' ', text).strip()

def split_text_into_chunks(text: str, chunk_size: int = 800, chunk_overlap: int = 100):
    if not text: return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

async def scrape_website_text(url: str):
    try:
        async with httpx.AsyncClient(verify=False, timeout=20.0) as client:
            response = await client.get(url)
            response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style"]): script.decompose()
        text = soup.get_text()
        return re.sub(r'\s+', ' ', text).strip()
    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch URL: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing website content: {e}")


# --- AIロジック関連のヘルパー関数 ---
def format_urls_as_links(text: str) -> str:
    url_pattern = r'(https?://[^\s\[\]()]+)'
    md_link_pattern = r'\[([^\]]+)\]\((https?://[^\s\)]+)\)'
    text = re.sub(url_pattern, r'<a href="\1" target="_blank">\1</a>', text)
    return re.sub(md_link_pattern, r'<a href="\2" target="_blank">\1</a>', text)


async def classify_query_intent(query: str, model: str) -> str:
    prompt = f"""ユーザーからの入力が、「学内情報に関する質問」か「一般的な雑談」かを分類してください。
- 学内情報（履修、施設、奨学金など）に関する質問 -> "学内情報"
- 挨拶、天気、日常会話、AI自身への質問など -> "雑談"
入力: "{query}"\n分類結果:"""
    try:
        gemini_model = genai.GenerativeModel(model)
        response = await gemini_model.generate_content_async(prompt)
        return "雑談" if "雑談" in response.text.strip() else "学内情報"
    except Exception: return "学内情報"

async def summarize_url_content(url: str, model: str) -> Optional[str]:
    try:
        text = await scrape_website_text(url)
        if not text or len(text) < 100: return None
        prompt = f"以下の文章は大学の公式サイトからの抜粋です。学生が知りたいであろう要点を3つ程度に絞り、箇条書きで簡潔に要約してください。\n# 文章\n{text[:8000]}\n# 要約:"
        gemini_model = genai.GenerativeModel(model)
        response = await gemini_model.generate_content_async(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"URLの要約中にエラー: {e}")
        return None

async def create_tiered_response(category: str, model: str):
    info = CATEGORY_INFO.get(category)
    if not info: return "申し訳ありませんが、お尋ねの件について情報が見つかりませんでした。[大学公式サイト](https://www.sgu.ac.jp/)をご確認ください。"
    
    summary = await summarize_url_content(info['url_to_summarize'], model) if 'url_to_summarize' in info else None
    if summary:
        return f"**▼ {category}に関する公式サイトの要約**\n{summary}\n\nより詳しい情報は、以下のリンクから直接ご確認ください。\n[{category}関連ページ]({info['url_to_summarize']})"
    return info.get("static_response", "情報が見つかりませんでした。")


# --------------------------------------------------------------------------
# 5. APIエンドポイント定義
# --------------------------------------------------------------------------

# --- 認証とHTML提供 ---
# --- 認証とHTML提供 ---
@app.get('/login')
async def login(request: Request):
    if 'auth0' not in oauth._clients: raise HTTPException(status_code=500, detail="Auth0 is not configured.")
    return await oauth.auth0.authorize_redirect(request, request.url_for('auth'))

@app.get('/auth')
async def auth(request: Request):
    if 'auth0' not in oauth._clients: raise HTTPException(status_code=500, detail="Auth0 is not configured.")
    token = await oauth.auth0.authorize_access_token(request)
    if userinfo := token.get('userinfo'): request.session['user'] = dict(userinfo)
    return RedirectResponse(url='/admin')

@app.get('/logout')
async def logout(request: Request):
    request.session.pop('user', None)
    if not all([AUTH0_DOMAIN, AUTH0_CLIENT_ID]): return RedirectResponse(url='/')
    return RedirectResponse(f"https://{AUTH0_DOMAIN}/v2/logout?returnTo={request.url_for('serve_client')}&client_id={AUTH0_CLIENT_ID}")

def require_sgu_member(request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=307, headers={'Location': '/login'})
    email = user.get('email', '')
    if email.endswith('@sgu.ac.jp') or email == "ishikawamasahito3150@gmail.com": return user
    raise HTTPException(status_code=403, detail="アクセス権がありません。")

@app.get("/", response_class=FileResponse)
async def serve_client(): return FileResponse(os.path.join(BASE_DIR, "client.html"))
@app.get("/admin", response_class=FileResponse)
async def serve_admin(_: dict = Depends(require_sgu_member)): return FileResponse(os.path.join(BASE_DIR, "admin.html"))
@app.get("/log", response_class=FileResponse)
async def serve_log_page(_: dict = Depends(require_sgu_member)): return FileResponse(os.path.join(BASE_DIR, "log.html"))
@app.get("/favicon.ico", include_in_schema=False)
async def favicon(): return Response(status_code=204)

# --- ステータス確認 ---
@app.get("/health")
async def health_check(): return {"status": "ok"}
@app.get("/chromadb/status", dependencies=[Depends(require_sgu_member)])
async def chromadb_status():
    if not db_client: raise HTTPException(status_code=500, detail="DB not initialized")
    try: db_client.heartbeat(); return {"status": "ok"}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
@app.get("/gemini/status", dependencies=[Depends(require_sgu_member)])
async def gemini_status():
    if GEMINI_API_KEY: return {"connected": True, "models": ["gemini-1.5-flash-latest", "gemini-pro"]}
    return {"connected": False, "detail": "GEMINI_API_KEY not set."}

# --- DBコレクション管理 ---
@app.get("/collections", dependencies=[Depends(require_sgu_member)])
async def list_collections():
    if not db_client: raise HTTPException(status_code=500, detail="ChromaDB client not initialized")
    collections = await asyncio.to_thread(db_client.list_collections)
    return [{"name": c.name, "count": c.count()} for c in collections]

@app.post("/collections", dependencies=[Depends(require_sgu_member)])
async def create_collection_api(req: CreateCollectionRequest):
    if not db_client: raise HTTPException(status_code=500, detail="ChromaDB client not initialized")
    if not (name := req.name.strip()): raise HTTPException(status_code=400, detail="Collection name is empty")
    try:
        collection = await asyncio.to_thread(db_client.create_collection, name=name)
        return {"name": collection.name, "message": "Collection created"}
    except Exception as e: raise HTTPException(status_code=500, detail=f"Failed to create collection: {e}")

@app.delete("/collections/{collection_name}", dependencies=[Depends(require_sgu_member)])
async def delete_collection_api(collection_name: str):
    if not db_client: raise HTTPException(status_code=500, detail="ChromaDB client not initialized")
    try:
        await asyncio.to_thread(db_client.delete_collection, name=collection_name)
        return {"message": "Collection deleted"}
    except Exception as e: raise HTTPException(status_code=500, detail=f"Failed to delete collection: {e}")

@app.get("/collections/{collection_name}/documents", dependencies=[Depends(require_sgu_member)])
async def get_documents(collection_name: str):
    if not db_client: raise HTTPException(status_code=500, detail="ChromaDB not available")
    try:
        collection = await asyncio.to_thread(db_client.get_collection, name=collection_name)
        if collection.count() == 0: return {"documents": [], "count": 0}
        data = await asyncio.to_thread(collection.get, include=['metadatas'])
        sources = sorted(list(set(md.get('filename') or md.get('source_url', 'unknown') for md in data['metadatas'])))
        return {"documents": [{"id": src} for src in sources], "count": collection.count()}
    except ValueError: return {"documents": [], "count": 0}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

# --- データ登録 (ファイル & URL) ---
async def add_text_to_collection(text: str, collection_name: str, embedding_model: str, metadata: dict):
    if not db_client:
        raise HTTPException(status_code=500, detail="DB client not ready")
    
    chunks = split_text_into_chunks(text)
    if not chunks:
        return 0

    total_chunks_added = 0
    batch_size = 50

    collection = await asyncio.to_thread(db_client.get_or_create_collection, name=collection_name)

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        
        result = await asyncio.to_thread(
            genai.embed_content,
            model=f"models/{embedding_model}",
            content=batch_chunks,
            task_type="retrieval_document"
        )
        embeddings = result['embedding']

        await asyncio.to_thread(
            collection.add,
            embeddings=embeddings,
            documents=batch_chunks,
            ids=[str(uuid.uuid4()) for _ in batch_chunks],
            metadatas=[metadata for _ in batch_chunks]
        )
        total_chunks_added += len(batch_chunks)
        await asyncio.sleep(35)

    return total_chunks_added

@app.post("/upload", dependencies=[Depends(require_sgu_member)])
async def upload_file_api(file: UploadFile = File(...), collection_name: str = Form(...), embedding_model: str = Form(...)):
    try:
        text = await read_file_content(file)
        chunks_added = await add_text_to_collection(text, collection_name, embedding_model, {"filename": file.filename})
        return {"message": f"File '{file.filename}' added", "chunks": chunks_added}
    except Exception as e:
        print(f"Upload failed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"ファイル処理中にエラーが発生しました: {str(e)}")


@app.post("/scrape", dependencies=[Depends(require_sgu_member)])
async def scrape_website_api(req: ScrapeRequest):
    try:
        text = await scrape_website_text(req.url)
        chunks_added = await add_text_to_collection(text, req.collection_name, req.embedding_model, {"source_url": req.url})
        return {"message": f"URL '{req.url}' scraped", "chunks": chunks_added}
    except Exception as e:
        print(f"Scrape failed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Webサイトの処理中にエラーが発生しました: {str(e)}")


# --- 設定とフィードバック ---
@app.post("/settings", dependencies=[Depends(require_sgu_member)])
async def update_settings_api(req: SettingsUpdateRequest):
    if not settings_manager: raise HTTPException(503, "Settings not initialized")
    await settings_manager.update_settings(req.dict())
    return {"message": "Settings updated"}

@app.get("/logs", dependencies=[Depends(require_sgu_member)])
async def get_logs_api():
    if not os.path.exists(FEEDBACK_FILE_PATH): return []
    try: return pd.read_csv(FEEDBACK_FILE_PATH).fillna("").to_dict('records')
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/feedback")
async def handle_feedback_api(req: FeedbackRequest):
    if not os.path.exists(FEEDBACK_FILE_PATH): raise HTTPException(404, "Log file not found")
    try:
        df = pd.read_csv(FEEDBACK_FILE_PATH)
        if req.log_id in df['log_id'].values:
            df.loc[df['log_id'] == req.log_id, 'rating'] = req.rating
            df.to_csv(FEEDBACK_FILE_PATH, index=False)
            return {"message": "Feedback received"}
        raise HTTPException(status_code=404, detail="Log ID not found")
    except Exception as e: raise HTTPException(500, str(e))

# --- WebSocket ---
@app.websocket("/ws/settings")
async def websocket_settings_endpoint(websocket: WebSocket):
    session_cookie = websocket.cookies.get("session")
    if not session_cookie or not APP_SECRET_KEY: await websocket.close(1008); return
    if not settings_manager: await websocket.close(1011); return
    await settings_manager.add_websocket(websocket)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect: settings_manager.remove_websocket(websocket)


# --- チャットAPI (統合ロジック) ---
async def enhanced_chat_logic(req: ChatRequest):
    log_id = str(uuid.uuid4())
    full_response, category, has_specific_info = "", "その他", False
    
    try:
        yield f"data: {json.dumps({'log_id': log_id})}\n\n"
        if not all([db_client, GEMINI_API_KEY]): raise HTTPException(503, "サービスが利用できません。")

        intent = await classify_query_intent(req.query, req.model)
        model = genai.GenerativeModel(req.model)

        if intent == "雑談":
            category = "雑談"
            prompt = f"あなたは札幌学院大学の親しみやすいAIアシスタントです。ユーザーから雑談をもちかけられています。フレンドリーかつ簡潔な日本語で、自然な会話をしてください。\nユーザー: {req.query}\nあなた:"
            stream = await model.generate_content_async(prompt, stream=True)
            async for chunk in stream:
                if chunk.text:
                    full_response += chunk.text
                    yield f"data: {json.dumps({'content': chunk.text})}\n\n"
            return

        category = next((cat for cat, keys in KEYWORD_MAP.items() if any(key in req.query for key in keys)), "その他")
        
        context = ""
        try:
            collection = await asyncio.to_thread(db_client.get_collection, name=req.collection)
            result = await asyncio.to_thread(genai.embed_content, model=f"models/{req.embedding_model}", content=[req.query], task_type="retrieval_query")
            
            results = await asyncio.to_thread(collection.query, query_embeddings=result['embedding'], n_results=req.top_k)
            if results['documents'] and results['documents'][0]:
                context = "\n".join(results['documents'][0])
                has_specific_info = True
        except Exception as e:
            print(f"DB検索エラー: {e}")

        if has_specific_info:
            prompt = f"あなたは札幌学院大学の学生を支援するAIです。以下の情報を元に、質問に日本語で回答してください。\n情報: {context}\n質問: {req.query}\n回答:"
            stream = await model.generate_content_async(prompt, stream=True)
            async for chunk in stream:
                if chunk.text:
                    full_response += chunk.text
                    yield f"data: {json.dumps({'content': chunk.text})}\n\n"
        else:
            tiered_response = await create_tiered_response(category, req.model)
            full_response = tiered_response
            yield f"data: {json.dumps({'content': full_response})}\n\n"

    except Exception as e:
        full_response = f"エラーが発生しました: {str(e)}"
        yield f"data: {json.dumps({'content': full_response})}\n\n"
        traceback.print_exc()
    
    finally:
        full_response = format_urls_as_links(full_response)
        try:
            with open(FEEDBACK_FILE_PATH, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([log_id, datetime.now(JST).isoformat(), req.query, full_response, "", category, has_specific_info])
        except Exception as e:
            print(f"ログ保存失敗: {e}")

@app.post("/chat", dependencies=[Depends(require_sgu_member)])
async def admin_chat(req: ChatRequest):
    # 管理画面から受け取った req をそのまま渡すのが正しい
    return StreamingResponse(enhanced_chat_logic(req), media_type="text/event-stream")

@app.post("/chat_for_client")
async def client_chat(req: ClientChatRequest):
    if not settings_manager: raise HTTPException(503, "Settings not initialized.")
    # ユーザーの質問(req)とサーバー設定を合体させた chat_request を作成
    chat_request = ChatRequest(query=req.query, **settings_manager.settings)
    # 新しく作った chat_request を渡すのが正しい
    return StreamingResponse(enhanced_chat_logic(chat_request), media_type="text/event-stream")

# --------------------------------------------------------------------------
# 6. 開発用サーバー起動
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("=== 札幌学院大学 学生サポートAI (ローカル開発モード) ===")
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)

