# --------------------------------------------------------------------------
# 1. ライブラリのインポート
# --------------------------------------------------------------------------
import os
import json
import httpx
import chromadb
import traceback
import csv
from datetime import datetime
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

from starlette.middleware.sessions import SessionMiddleware
from starlette.config import Config
from authlib.integrations.starlette_client import OAuth
from fastapi import Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from bs4 import BeautifulSoup
# --- Prometheusのインポートを追加 ---
from prometheus_fastapi_instrumentator import Instrumentator

# .env 読み込み（必要なら）
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --------------------------------------------------------------------------
# 2. FastAPIアプリケーションの初期設定
# --------------------------------------------------------------------------
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Gemini APIキーの設定 ---
try:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("環境変数 'GEMINI_API_KEY' が設定されていません。")
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini APIキーが正常に設定されました。")
except Exception as e:
    print(f"重大なエラー: Gemini APIキーの設定に失敗しました。 - {e}")
    GEMINI_API_KEY = None

# --- Auth0/アプリのシークレット読み込み ---
AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
# セッション用シークレット（どちらか使う想定）
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY") or os.getenv("SESSION_SECRET")

def _mask(value: Optional[str], show: int = 4) -> str:
    if not value:
        return "未設定"
    if len(value) <= show:
        return "*" * len(value)
    return value[:show] + "..." + "*" * max(0, len(value) - show - 3)

print("=" * 50)
print("Auth0設定確認:")
print(f"CLIENT_ID: {AUTH0_CLIENT_ID is not None}")
print(f"CLIENT_ID値(一部): {_mask(AUTH0_CLIENT_ID)}")
print(f"DOMAIN: {AUTH0_DOMAIN or '未設定'}")
print(f"SECRET_KEY: {APP_SECRET_KEY is not None}")
print(f"SECRET_KEY長: {len(APP_SECRET_KEY) if APP_SECRET_KEY else 0}文字")
print("=" * 50)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # --- パス設定 ---

# Renderの永続ディスクを使用するためのパス設定
RENDER_DISK_PATH = "/var/data"
# Render環境かどうかを永続ディスクの存在で判定
IS_RENDER_ENV = os.path.exists(RENDER_DISK_PATH)

if IS_RENDER_ENV:
    DB_PATH = os.path.join(RENDER_DISK_PATH, "chroma_db")
    FEEDBACK_FILE_PATH = os.path.join(RENDER_DISK_PATH, "feedback.csv")
else:
    DB_PATH = os.path.join(BASE_DIR, "chroma_db")
    FEEDBACK_FILE_PATH = os.path.join(BASE_DIR, "feedback.csv")

SETTINGS_FILE_PATH = os.path.join(BASE_DIR, "shared_settings.json")

# --- グローバル設定管理 ---
class SettingsManager:
    def __init__(self):
        self.settings = {
            "model": "gemini-1.5-flash-latest",
            "collection": "default",
            "embedding_model": "text-embedding-004",
            "top_k": 5
        }
        self.websocket_connections: Dict[str, WebSocket] = {}
        self.load_settings()

    def load_settings(self):
        try:
            if os.path.exists(SETTINGS_FILE_PATH):
                with open(SETTINGS_FILE_PATH, 'r', encoding='utf-8') as f:
                    saved_settings = json.load(f)
                    self.settings.update(saved_settings)
                    print(f"設定をロードしました: {self.settings}")
        except Exception as e:
            print(f"設定ファイルの読み込みエラー: {e}")

    def save_settings(self):
        try:
            with open(SETTINGS_FILE_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)
                print(f"設定を保存しました: {self.settings}")
        except Exception as e:
            print(f"設定ファイルの保存エラー: {e}")

    def update_settings(self, new_settings: Dict[str, Any]):
        self.settings.update(new_settings)
        self.save_settings()
        asyncio.create_task(self.broadcast_settings())

    async def add_websocket(self, connection_id: str, websocket: WebSocket):
        self.websocket_connections[connection_id] = websocket
        try:
            await websocket.send_json({"type": "settings_update", "data": self.settings})
        except Exception as e:
            print(f"初期設定送信エラー: {e}")

    def remove_websocket(self, connection_id: str):
        if connection_id in self.websocket_connections:
            del self.websocket_connections[connection_id]

    async def broadcast_settings(self):
        message = {"type": "settings_update", "data": self.settings}
        disconnected = [cid for cid, ws in self.websocket_connections.items() if ws.client_state.name != 'CONNECTED']
        for cid in disconnected: self.remove_websocket(cid)

        active_connections = list(self.websocket_connections.values())
        tasks = [conn.send_json(message) for conn in active_connections]
        await asyncio.gather(*tasks, return_exceptions=True)

settings_manager = SettingsManager()

# --- カテゴリマッピング ---
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
    "授業": {
        "url_to_summarize": "https://www.sgu.ac.jp/information/schedule.html",
        "static_response": """学事暦や年間スケジュールに関するご質問ですね。
- **夏期休業期間（夏休み）**: 8月上旬～9月中旬
- **冬季休業期間（冬休み）**: 12月下旬～1月上旬
- **春季休業期間（春休み）**: 2月上旬～3月下旬
正確な日付は、以下の公式でご確認ください。
- **[2025年度 学事暦](https://www.sgu.ac.jp/information/schedule.html**
"""
    },
    "証明書": {
        "url_to_summarize": "https://www.sgu.ac.jp/campuslife/support/certification.html",
        "static_response": "各種証明書の発行については、[諸証明・各種願出・届出について](https://www.sgu.ac.jp/campuslife/support/certification.html)のページをご確認ください。"
    },
    "経済支援": {
        "url_to_summarize": "https://www.sgu.ac.jp/campuslife/scholarship/",
        "static_response": "奨学金や授業料減免については、[奨学金制度](https://www.sgu.ac.jp/campuslife/scholarship/)や[授業料減免制度](https://www.sgu.ac.jp/tuition/j09tjo00000f665g.html)のページをご確認ください。"
    },
    "留学": {
        "url_to_summarize": "https://www.sgu.ac.jp/abroad/",
        "static_response": "留学や国際交流については、[留学・国際交流](https://www.sgu.ac.jp/abroad/)のページをご確認ください。"
    },
    "生協": {
        "url_to_summarize": "https://www.hokkaido-univcoop.jp/sgu/",
        "static_response": "生協に関するご質問ですね。食事や書籍、各種サービスなど、詳しくは[生協公式サイト](https://www.hokkaido-univcoop.jp/sgu/)をご確認ください。"
    },
    "施設": {
        "url_to_summarize": "https://www.sgu.ac.jp/information/ebetsu_campus2.html",
        "static_response": "キャンパス内の施設については[キャンパスマップ・施設紹介](https://www.sgu.ac.jp/information/ebetsu_campus2.html)や[図書館公式サイト](https://www.sgu.ac.jp/library/)をご確認ください。"
    },
    "カリキュラム": {
        "url_to_summarize": "https://www.sgu.ac.jp/faculty/",
        "static_response": "カリキュラムについては、[学部・大学院](https://www.sgu.ac.jp/faculty/)のページからご自身の所属する学部・学科の情報をご確認ください。"
    },
    "相談": {
        "url_to_summarize": "https://www.sgu.ac.jp/campuslife/counseling-room.html",
        "static_response": "心や学生生活の悩みについては、[学生相談室](https://www.sgu.ac.jp/campuslife/counseling-room.html)のほか、**保健センター(011-375-8501)**や**サポートセンター(011-375-8567)**などの相談窓口をご利用ください。"
    },
    "欠席手続き": {
        "url_to_summarize": "https://www.sgu.ac.jp/campuslife/support/j09tjo000003g3l6.html",
        "static_response": "授業の欠席に関する手続きは、[諸証明 / 各種願出・届出](https://www.sgu.ac.jp/campuslife/support/j09tjo000003g3l6.html)のページをご確認いただくか、所属学部の教育支援課にご相談ください。"
    },
    "試験": {
        "url_to_summarize": "https://www.sgu.ac.jp/campuslife/class/",
        "static_response": "試験や成績に関する一般的な情報は、[授業・成績](https://www.sgu.ac.jp/campuslife/class/)のページをご確認ください。詳細については、シラバスや所属学部の教育支援課にお問い合わせください。"
    },
    "学生支援": {
        "url_to_summarize": "https://www.sgu.ac.jp/campuslife/",
        "static_response": "学生生活全般に関するサポートについては、[キャンパスライフ](https://www.sgu.ac.jp/campuslife/)のページをご覧ください。サークル活動や奨学金など、様々な情報が掲載されています。"
    },
    "資格": {
        "url_to_summarize": "https://www.sgu.ac.jp/qualification/",
        "static_response": "資格取得や各種講座については、[資格・課程](https://www.sgu.ac.jp/qualification/)のページで詳細をご確認ください。"
    }
}

db_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_client
    print("--- アプリケーション起動処理開始 ---")
    try:
        if not os.path.exists(FEEDBACK_FILE_PATH):
            with open(FEEDBACK_FILE_PATH, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(['log_id', 'timestamp', 'rating', 'query', 'response', 'category', 'has_specific_info'])
        db_client = chromadb.PersistentClient(path=DB_PATH)
    except Exception as e:
        print(f"致命的エラー: {e}")
    
    # --- ★★★ FINAL DEBUG: Print all registered routes on startup ★★★ ---
    print("\n--- REGISTERED ROUTES ---")
    for route in app.routes:
        if hasattr(route, "path"):
            methods = getattr(route, 'methods', 'N/A')
            print(f"Path: {route.path} | Methods: {methods} | Name: {route.name}")
    print("--- END REGISTERED ROUTES ---\n")
    # --- ★★★ END FINAL DEBUG ★★★ ---

    print("--- 起動処理完了 ---")
    yield
    print("--- アプリケーション終了処理開始 ---")
    # --- Auth0設定 ---
# この部分は後で作成する.envファイルから読み込まれます
config = Config('.env')
AUTH0_CLIENT_ID = config('AUTH0_CLIENT_ID', cast=str, default=None)
AUTH0_CLIENT_SECRET = config('AUTH0_CLIENT_SECRET', cast=str, default=None)
AUTH0_DOMAIN = config('AUTH0_DOMAIN', cast=str, default=None)
APP_SECRET_KEY = config('APP_SECRET_KEY', cast=str, default=None)

# OAuthクライアントの初期化
oauth = OAuth(config)
oauth.register(
    name='auth0',
    client_id=AUTH0_CLIENT_ID,
    client_secret=AUTH0_CLIENT_SECRET,
    server_metadata_url=f'https://{AUTH0_DOMAIN}/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid profile email',
    },
)

# セッション管理のためのミドルウェアを追加
# APP_SECRET_KEYはセッション情報を暗号化するための秘密鍵

# --- Auth0設定ここまで ---

app = FastAPI(lifespan=lifespan)

if APP_SECRET_KEY:
    app.add_middleware(SessionMiddleware, secret_key=APP_SECRET_KEY)


# --- Prometheusのメトリクスを有効にするためのコードを追加 ---
Instrumentator().instrument(app).expose(app)

# --------------------------------------------------------------------------
# 3. ヘルパー関数とPydanticモデル定義
# --------------------------------------------------------------------------
async def get_gemini_embedding(text_chunks: List[str], model_name: str = "text-embedding-004") -> List[List[float]]:
    """Gemini APIを使用してテキストの埋め込みを取得する"""
    if not GEMINI_API_KEY:
        raise ValueError("Gemini APIキーが設定されていません。")
    try:
        result = await asyncio.to_thread(
            genai.embed_content,
            model=f"models/{model_name}",
            content=text_chunks,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        print(f"Gemini Embedding APIエラー: {e}")
        traceback.print_exc()
        return [[] for _ in text_chunks]

def format_urls_as_links(text: str) -> str:
    """テキスト内のURLをHTMLのaタグに変換する"""
    # 既にaタグになっている部分を保護
    protected_text = re.sub(r'(<a.*?href=".*?".*?>.*?<\/a>)', r'__PROTECTED_LINK__\1__END_PROTECTED_LINK__', text)
    
    # 素のURLを a タグに変換
    url_pattern = r'(https?://[^\s\[\]()]+)'
    text_with_raw_urls = re.sub(url_pattern, lambda m: f'<a href="{m.group(1).rstrip(".,;:!?")}" target="_blank" rel="noopener noreferrer">{m.group(1).rstrip(".,;:!?")}</a>', protected_text)

    # Markdown形式のリンク [text](url) を a タグに変換
    md_link_pattern = r'\[([^\]]+)\]\((https?://[^\s\)]+)\)'
    final_text = re.sub(md_link_pattern, r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>', text_with_raw_urls)

    # 保護した部分を元に戻す
    final_text = final_text.replace('__PROTECTED_LINK__', '').replace('__END_PROTECTED_LINK__', '')
    
    return final_text


def create_classification_prompt(query: str) -> str:
    """ユーザーの質問の意図を分類するためのプロンプトを生成する"""
    return f"""ユーザーからの入力が、「学内情報に関する質問」か「一般的な雑談」かを分類してください。
    - 学内情報（履修、施設、奨学金など）に関する質問 -> "学内情報"
    - 挨拶、天気、日常会話、AI自身への質問など -> "雑談"

    入力: "{query}"
    分類結果:"""

def create_chitchat_prompt(query: str) -> str:
    """雑談用のプロンプトを生成する"""
    return f"""# 指示
あなたは札幌学院大学の学生をサポートする、親しみやすいAIアシスタントです。ユーザーから雑談をもちかけられています。フレンドリーかつ簡潔な日本語で、自然な会話をしてください。
# ユーザーの言葉: {query}
# あなたの応答:"""

async def classify_query_intent(query: str, model: str) -> str:
    """AIを使ってユーザーの質問の意図を分類する"""
    try:
        gemini_model = genai.GenerativeModel(model)
        prompt = create_classification_prompt(query)
        response = await gemini_model.generate_content_async(prompt)
        # 応答テキストから "雑談" または "学内情報" のみを抽出
        classification = response.text.strip()
        if "雑談" in classification:
            return "雑談"
        return "学内情報"
    except Exception as e:
        print(f"意図分類エラー: {e}")
        return "学内情報" # エラー時は安全策として学内情報として処理

def create_summarization_prompt(text: str) -> str:
    """URL要約用のプロンプトを生成する"""
    return f"""# 指示
以下の文章は大学の公式サイトからの抜粋です。学生が知りたいであろう要点を3つ程度に絞り、箇条書きで簡潔に要約してください。
# 文章
{text}
# 要約:"""

async def summarize_url_content(url: str, model: str) -> Optional[str]:
    """指定されたURLのコンテンツを取得し、Geminiで要約する"""
    print(f"URLの要約を開始: {url}")
    if not GEMINI_API_KEY:
        return "APIキーが設定されていないため、要約できません。"
    try:
        async with httpx.AsyncClient(timeout=15, verify=False) as client:
            response = await client.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()

        soup = BeautifulSoup(response.text, 'lxml')
        for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            tag.decompose()

        text = soup.get_text(separator='\n', strip=True)
        if not text or len(text) < 100:
            return None

        summary_prompt = create_summarization_prompt(text[:8000]) # コンテキスト長を考慮
        gemini_model = genai.GenerativeModel(model)
        summary_response = await gemini_model.generate_content_async(summary_prompt)

        return summary_response.text.strip()
    except Exception as e:
        print(f"URLの要約中にエラーが発生しました: {e}")
        return None

def get_static_fallback_message() -> str:
    """情報が見つからなかった場合の静的なフォールバックメッセージを返す"""
    fallback_response = f"""申し訳ありませんが、お尋ねの件について詳細な情報が見つかりませんでした。
正確な情報については、[大学公式サイト](https://www.sgu.ac.jp/)をご確認いただくか、適切な窓口へお問い合わせください。
**大学代表**: 011-386-8111"""
    return format_urls_as_links(fallback_response)

async def create_tiered_response_system(category: str, has_specific_info: bool, model: str) -> str:
    """段階的な応答を生成するシステム"""
    if not has_specific_info and category in CATEGORY_INFO:
        info = CATEGORY_INFO[category]
        url_to_summarize = info.get("url_to_summarize")
        static_response = info.get("static_response", get_static_fallback_message())

        if url_to_summarize:
            summary = await summarize_url_content(url_to_summarize, model)
            if summary:
                response_text = f"**▼ {category}に関する公式サイトの要約**\n{summary}\n\nより詳しい情報は、以下のリンクから直接ご確認ください。\n[{category}関連ページ]({url_to_summarize})"
                if category == "授業":
                    pdf_links_info = """\n\n---\n\nまた、詳細な日程については、以下の公式PDFをご確認いただくのが確実です。\n\n- **[2025年度 札幌学院大学 学事暦（PDF）](https://www.sgu.ac.jp/information/j09tjo00000el5w6-att/j09tjo00000giz0l.pdf)**"""
                    response_text += pdf_links_info
                return format_urls_as_links(response_text)
        return format_urls_as_links(static_response)
    return get_static_fallback_message()

def create_enhanced_response_prompt(query: str, context: str, category: str) -> str:
    """RAG用のプロンプトを生成する"""
    return f"""# システム指示
あなたは札幌学院大学の学生サポートAIです。学内資料に基づいて正確な情報のみを提供してください。
## 絶対遵守ルール
1. **情報源の制限**: 提供された参考情報にない内容は絶対に回答しない
2. **推測の禁止**: 「〜と思われます」「〜かもしれません」等の推測表現は使用禁止
## 参考情報
{context}
## 学生からの質問
{query}
## 回答（参考情報に基づく事実のみ）:"""

def verify_response_accuracy(response: str, context: str) -> dict:
    """応答の正確性を検証する（簡易版）"""
    verification_result = {"is_accurate": True, "issues": []}
    speculation_patterns = [r'と思います', r'と思われます', r'かもしれません', r'でしょう', r'一般的に', r'おそらく']
    if any(re.search(p, response) for p in speculation_patterns):
        verification_result["is_accurate"] = False
        verification_result["issues"].append("推測的な表現が含まれています。")
    return verification_result

class ScrapeRequest(BaseModel):
    url: str
    collection_name: str
    embedding_model: str = "text-embedding-004"

class FeedbackRequest(BaseModel):
    log_id: str
    rating: str

class CollectionRequest(BaseModel):
    name: str

class ChatRequest(BaseModel):
    query: str
    model: str = "gemini-1.5-flash-latest"
    collection: str = "default"
    embedding_model: str = "text-embedding-004"
    top_k: int = 3

class ClientChatRequest(BaseModel):
    query: str

class SettingsUpdateRequest(BaseModel):
    model: Optional[str] = None
    collection: Optional[str] = None
    embedding_model: Optional[str] = None
    top_k: Optional[int] = None

# --------------------------------------------------------------------------
# 4. APIエンドポイント定義
# --------------------------------------------------------------------------
@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_client():
    file_path = os.path.join(BASE_DIR, "client.html")
    
    # --- デバッグ用ログ ---
    print(f"--- CLIENT ROUTE ---")
    print(f"BASE_DIR is: {BASE_DIR}")
    print(f"Attempting to access file at: {file_path}")
    print(f"Does file exist? -> {os.path.exists(file_path)}")
    # --- ここまで ---

    if not os.path.exists(file_path): 
        raise HTTPException(status_code=404, detail=f"client.html not found at path: {file_path}")
    return FileResponse(file_path)
# --- 認証エンドポイント ---
@app.get('/login')
async def login(request: Request):
    """ユーザーをAuth0のログインページにリダイレクトする"""
    redirect_uri = request.url_for('auth')
    return await oauth.auth0.authorize_redirect(request, redirect_uri)

@app.get('/auth')
async def auth(request: Request):
    """Auth0からのコールバックを処理し、セッションにユーザー情報を保存する"""
    token = await oauth.auth0.authorize_access_token(request)
    user_info = token.get('userinfo')
    if user_info:
        request.session['user'] = dict(user_info)
    return RedirectResponse(url='/')

@app.get('/logout')
async def logout(request: Request):
    """ユーザーセッションをクリアし、Auth0からログアウトさせる"""
    request.session.pop('user', None)
    logout_url = f"https://{AUTH0_DOMAIN}/v2/logout?" + \
                 f"returnTo={request.url_for('serve_client')}&" + \
                 f"client_id={AUTH0_CLIENT_ID}"
    return RedirectResponse(url=logout_url)

# --- 認証状態を確認するためのヘルパー関数 ---
def get_current_user(request: Request) -> Optional[dict]:
    """セッションから現在のユーザー情報を取得する"""
    return request.session.get('user')

def is_sgu_member(user: Optional[dict] = Depends(get_current_user)) -> bool:
    """ユーザーが大学のメンバーか、あるいは管理者かを確認する"""
    if not user:
        return False
    
    email = user.get('email', '')
    
    # ★★★ 管理者であるあなたの個人メールアドレスをここに追加 ★★★
    admin_emails = ["ishikawamasahito3150@gmail.com"] 
    
    # 条件1: メールアドレスが管理者リストにあれば、無条件でアクセスを許可
    if email in admin_emails:
        return True
        
    # 条件2: それ以外の場合は、大学のドメインをチェック
    return email.endswith('@sgu.ac.jp')
# --- 認証エンドポイントここまで ---


@app.get("/admin", response_class=FileResponse, include_in_schema=False)
async def serve_admin(request: Request, is_member: bool = Depends(is_sgu_member)):
    """大学メンバーのみがアクセスできる保護された管理ページ"""
    if not is_member:
        # ログインしていない、または大学のメンバーでない場合はログインページへリダイレクト
        return RedirectResponse(url='/login')
    
    file_path = os.path.join(BASE_DIR, "admin.html")
    if not os.path.exists(file_path): 
        raise HTTPException(status_code=404, detail="admin.html not found.")
    return FileResponse(file_path)



@app.get("/log", response_class=FileResponse, include_in_schema=False)
async def serve_log_page():
    file_path = os.path.join(BASE_DIR, "log.html")
    if not os.path.exists(file_path): raise HTTPException(status_code=404, detail="log.html not found.")
    return FileResponse(file_path)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.websocket("/ws/settings")
async def websocket_settings_endpoint(websocket: WebSocket):
    connection_id = str(uuid.uuid4())
    await websocket.accept()
    await settings_manager.add_websocket(connection_id, websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        print(f"WebSocket切断: {connection_id}")
    finally:
        settings_manager.remove_websocket(connection_id)

@app.post("/chat_for_client")
async def client_chat(req: ClientChatRequest):
    current_settings = settings_manager.settings
    chat_request = ChatRequest(query=req.query, **current_settings)
    return await enhanced_chat(chat_request)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/settings")
async def get_settings():
    return settings_manager.settings

@app.post("/settings")
async def update_settings_endpoint(request: SettingsUpdateRequest):
    update_data = request.dict(exclude_unset=True)
    if update_data:
        settings_manager.update_settings(update_data)
    return {"status": "success", "updated_settings": update_data}

@app.get("/chromadb/status")
async def chromadb_status():
    if not db_client: raise HTTPException(status_code=500, detail="ChromaDB client not initialized")
    try:
        db_client.heartbeat()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gemini/status")
async def gemini_status():
    if GEMINI_API_KEY:
        return {"connected": True, "models": ["gemini-1.5-flash-latest", "gemini-pro"]}
    else:
        return {"connected": False, "detail": "GEMINI_API_KEY environment variable not set."}

@app.get("/collections")
async def list_collections():
    if not db_client: raise HTTPException(status_code=500, detail="ChromaDB client not initialized")
    collections = await asyncio.to_thread(db_client.list_collections)
    return [{"name": c.name, "count": c.count()} for c in collections]

@app.post("/collections")
async def create_collection(req: CollectionRequest):
    if not db_client: raise HTTPException(status_code=500, detail="ChromaDB not available")
    await asyncio.to_thread(db_client.get_or_create_collection, name=req.name)
    settings_manager.update_settings({"collection": req.name})
    return JSONResponse(content={"message": f"Collection '{req.name}' created."}, status_code=201)

@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    if not db_client: raise HTTPException(status_code=500, detail="ChromaDB not available")
    await asyncio.to_thread(db_client.delete_collection, name=collection_name)
    if settings_manager.settings.get("collection") == collection_name:
        settings_manager.update_settings({"collection": "default"})
    return {"message": f"Collection '{collection_name}' deleted."}

@app.get("/collections/{collection_name}/documents")
async def get_documents_in_collection(collection_name: str):
    if not db_client: raise HTTPException(status_code=500, detail="ChromaDB not available")
    try:
        collection = await asyncio.to_thread(db_client.get_collection, name=collection_name)
        if collection.count() == 0: return {"documents": [], "count": 0}
        data = await asyncio.to_thread(collection.get, include=['metadatas'])
        sources = set(md.get('filename') or md.get('source_url', 'unknown') for md in data['metadatas'])
        return {"documents": [{"id": src} for src in sorted(list(sources))], "count": collection.count()}
    except ValueError:
        return {"documents": [], "count": 0}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), collection_name: str = Form(...), embedding_model: str = Form(...)):
    if not db_client: raise HTTPException(status_code=500, detail="ChromaDB not available")
    contents = await file.read()
    text = ""
    if file.filename.lower().endswith('.pdf'):
        reader = pypdf.PdfReader(io.BytesIO(contents))
        text = "".join(page.extract_text() or "" for page in reader.pages)
    elif file.filename.lower().endswith('.docx'):
        doc = docx.Document(io.BytesIO(contents))
        text = "\n".join(para.text for para in doc.paragraphs)
    else:
        try: text = contents.decode('utf-8')
        except UnicodeDecodeError: text = contents.decode('shift_jis', errors='ignore')

    if not text.strip(): raise HTTPException(status_code=400, detail="No text extracted.")

    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    embeddings = await get_gemini_embedding(chunks, embedding_model)

    collection = db_client.get_or_create_collection(name=collection_name)
    ids = [f"{file.filename}:{i}" for i in range(len(chunks))]
    category = file.filename.split('_', 1)[0] if '_' in file.filename else "general"
    metadatas = [{"filename": file.filename, "category": category} for _ in chunks]

    await asyncio.to_thread(collection.add, embeddings=embeddings, documents=chunks, metadatas=metadatas, ids=ids)
    return {"message": "File processed successfully", "chunks": len(chunks)}

@app.post("/scrape")
async def scrape_website(req: ScrapeRequest):
    if not db_client: raise HTTPException(status_code=500, detail="ChromaDB not available")
    async with httpx.AsyncClient(timeout=15, verify=False) as client:
        response = await client.get(req.url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(response.text, 'lxml')
    text = soup.get_text(separator='\n', strip=True)
    if not text: raise HTTPException(status_code=400, detail="No text content found.")

    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    embeddings = await get_gemini_embedding(chunks, req.embedding_model)

    collection = db_client.get_or_create_collection(name=req.collection_name)
    ids = [f"{req.url}:{i}" for i in range(len(chunks))]
    metadatas = [{"source_url": req.url, "category": "scraped"} for _ in chunks]

    await asyncio.to_thread(collection.add, embeddings=embeddings, documents=chunks, metadatas=metadatas, ids=ids)
    return {"message": f"Successfully added content from {req.url}", "chunks": len(chunks)}

@app.get("/logs")
async def get_logs():
    logs_data = []
    if not os.path.exists(FEEDBACK_FILE_PATH):
        return []
    try:
        with open(FEEDBACK_FILE_PATH, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                logs_data.append(row)
        return logs_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def handle_feedback(req: FeedbackRequest):
    if not os.path.exists(FEEDBACK_FILE_PATH):
        raise HTTPException(status_code=404, detail="Log file not found.")

    rows = []
    updated = False
    try:
        with open(FEEDBACK_FILE_PATH, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows.append(header)
            log_id_index = header.index('log_id')
            rating_index = header.index('rating')
            for row in reader:
                if row and row[log_id_index] == req.log_id:
                    row[rating_index] = req.rating
                    updated = True
                rows.append(row)

        if not updated:
            raise HTTPException(status_code=404, detail=f"Log ID {req.log_id} not found.")

        with open(FEEDBACK_FILE_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        return {"status": "success", "message": "Feedback updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------------------------------------------
# 5. 強化されたチャットエンドポイント (Gemini対応)
# --------------------------------------------------------------------------
@app.post("/chat")
async def enhanced_chat(req: ChatRequest):
    if not db_client: raise HTTPException(status_code=500, detail="DB not available")
    if not GEMINI_API_KEY: raise HTTPException(status_code=503, detail="Gemini API Key is not configured on the server.")

    log_id = str(uuid.uuid4())
    model_name = "gemini-1.5-flash-latest" if "flash" in req.model else req.model
    gemini_model = genai.GenerativeModel(model_name)

    async def generate_response():
        full_response = ""
        ai_classified_category = "その他"
        has_specific_info = False

        try:
            yield f"data: {json.dumps({'log_id': log_id})}\n\n"
            print(f"[{log_id}] チャット処理開始: {req.query}")

            # --- 意図分類 ---
            intent = await classify_query_intent(req.query, req.model)
            yield f"data: {json.dumps({'routing_result': intent})}\n\n"
            print(f"[{log_id}] 意図分類結果: {intent}")


            # --- 意図に応じた処理の分岐 ---
            if intent == "雑談":
                ai_classified_category = "雑談"
                chitchat_prompt = create_chitchat_prompt(req.query)
                response_stream = await gemini_model.generate_content_async(chitchat_prompt, stream=True)
                async for chunk in response_stream:
                    content = chunk.text
                    full_response += content
                    yield f"data: {json.dumps({'content': content})}\n\n"
                return

            # --- 学内情報の質問処理 ---
            vacation_keywords = ["夏休み", "冬休み", "春休み", "夏季休業", "冬季休業", "春季休業"]
            is_vacation_query = any(keyword in req.query for keyword in vacation_keywords)

            if is_vacation_query:
                print(f"[{log_id}] 長期休業に関する質問を検出。静的応答を返します。")
                ai_classified_category = "授業"
                yield f"data: {json.dumps({'routing_result': ai_classified_category})}\n\n"
                static_response = CATEGORY_INFO["授業"]["static_response"]
                formatted_response = format_urls_as_links(static_response)
                yield f"data: {json.dumps({'content': formatted_response, 'response_type': 'tiered'})}\n\n"
                full_response = formatted_response
                return

            matched_category = next((cat for cat, keys in KEYWORD_MAP.items() if any(key in req.query for key in keys)), "その他")
            ai_classified_category = matched_category
            print(f"[{log_id}] キーワード分類結果: '{ai_classified_category}'")
            yield f"data: {json.dumps({'routing_result': ai_classified_category})}\n\n"

            context = ""
            try:
                collection = db_client.get_collection(name=req.collection)
                query_embedding_response = await asyncio.to_thread(
                    genai.embed_content,
                    model=f"models/{req.embedding_model}",
                    content=req.query,
                    task_type="retrieval_query"
                )
                query_embedding = query_embedding_response['embedding']

                where_filter = {"category": ai_classified_category} if ai_classified_category != "その他" else None
                results = await asyncio.to_thread(
                    collection.query,
                    query_embeddings=[query_embedding],
                    n_results=req.top_k,
                    where=where_filter,
                    include=['documents']
                )

                if results and results.get('documents') and results['documents'][0]:
                    context = "\n---\n".join(results['documents'][0])
                    has_specific_info = True
                    yield f"data: {json.dumps({'search_results': True})}\n\n"
            except Exception as e:
                print(f"[{log_id}] DB検索エラー: {e}")

            if not has_specific_info:
                print(f"[{log_id}] 特定情報なし。段階的応答システムを実行。")
                tiered_response = await create_tiered_response_system(ai_classified_category, False, req.model)
                yield f"data: {json.dumps({'content': tiered_response, 'response_type': 'tiered'})}\n\n"
                full_response = tiered_response
                return

            enhanced_prompt = create_enhanced_response_prompt(req.query, context, ai_classified_category)
            response_stream = await gemini_model.generate_content_async(enhanced_prompt, stream=True)
            async for chunk in response_stream:
                content = chunk.text
                full_response += content
                yield f"data: {json.dumps({'content': content})}\n\n"

            full_response = format_urls_as_links(full_response)
            verification = verify_response_accuracy(full_response, context)
            if not verification["is_accurate"]:
                fallback_response = await create_tiered_response_system(ai_classified_category, False, req.model)
                fallback_message = f"\n\n**--- より確実な情報をご案内します ---**\n{fallback_response}"
                yield f"data: {json.dumps({'content': fallback_message, 'response_type': 'fallback'})}\n\n"
                full_response += fallback_message

        except Exception as e:
            print(f"[{log_id}] チャット処理全体でエラー: {e}")
            traceback.print_exc()
            error_response = get_static_fallback_message()
            yield f"data: {json.dumps({'content': error_response, 'response_type': 'error'})}\n\n"
            full_response = error_response

        finally:
            try:
                with open(FEEDBACK_FILE_PATH, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([log_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'N/A', req.query, full_response, ai_classified_category, has_specific_info])
            except Exception as e:
                print(f"[{log_id}] ログ記録エラー: {e}")

    return StreamingResponse(generate_response(), media_type='text/event-stream')

if __name__ == "__main__":
    import uvicorn
    print("=== 札幌学院大学 学生サポートAI (Gemini API版) ===")
    if not GEMINI_API_KEY:
        print("\n!!! 警告: 環境変数 'GEMINI_API_KEY' が設定されていません。チャット機能は動作しません。!!!\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
