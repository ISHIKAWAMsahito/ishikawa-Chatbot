# --------------------------------------------------------------------------
# 1. ライブラリのインポート
# --------------------------------------------------------------------------
import os
import json
import uvicorn
import traceback
import csv
from datetime import datetime, timezone, timedelta
import uuid
import io
import logging
import asyncio
import re
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

# --- サードパーティライブラリ ---
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv

from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse, FileResponse, Response
from pydantic import BaseModel

from supabase import create_client, Client
import requests
from bs4 import BeautifulSoup
import PyPDF2
from docx import Document as DocxDocument

from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth

# --- 初期設定 ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')
load_dotenv()

# --------------------------------------------------------------------------
# 2. 環境変数と基本設定
# --------------------------------------------------------------------------
# Gemini API設定
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY: 
    raise ValueError("環境変数 'GEMINI_API_KEY' が設定されていません。")
genai.configure(api_key=GEMINI_API_KEY)

# Auth0設定
AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "default-secret-key-for-development")

# Supabase設定
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# 定数
ACTIVE_COLLECTION_NAME = "student-knowledge-base"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JST = timezone(timedelta(hours=+9), 'JST')

# キーワードマッピング
KEYWORD_MAP = {
    # 授業関連
    "授業": [
        "学事暦", "シラバス", "授業計画", "年間スケジュール", "夏休み", "冬休み", "春休み",
        "夏季休業", "冬季休業", "春季休業", "授業時間", "90分", "講時", "補講", "休講",
        "情報ポータル", "授業", "カリキュラム"
        # 俗語・略語
        "コマ", "1限", "2限", "3限", "4限", "5限", "6限", "授サボ", "出サボ", "授サ",
        "オンデマ", "オンデマンド", "対面", "リモート", "ズーム授業"
    ],

    # 欠席関連
    "欠席手続き": [
        "忌引き", "休む", "欠席", "休講", "病気", "インフルエンザ", "欠席届", "公認欠席",
        "特別欠席", "感染症", "教育実習", "クラブ", "公式大会", "学校保健安全法",
        # 俗語・略語
        "サボる", "サボ", "バックれる", "体調不良", "風邪", "熱", "寝坊", "遅刻", "遅延証明"
    ],

    # 試験関連
    "試験": [
        "試験", "テスト", "追試", "再試", "成績", "レポート", "課題", "定期試験", "60分",
        "成績評価", "秀", "優", "良", "可", "不可", "GPA", "Grade Point Average",
        "成績確認", "不正行為", "カンニング", "学生証", "受験許可証",
        # 俗語・略語
        "単位落ち", "単落ち", "単位取る", "単位落とす", "赤点", "追試代", "再試代", "テス勉",
        "テスト勉強", "カンペ", "チート", "スマホ持ち込み", "時計代わり"
    ],

    # 証明書関連
    "証明書": [
        "証明書", "学割", "発行", "学生証", "休学", "復学", "退学", "再入学", "転学部",
        "転学科", "卒業", "在学", "卒業見込", "在学年限", "8年", "除籍", "卒業延期",
        "前期末卒業", "卒業見込証明書",
        # 俗語・略語
        "学割証", "学割切符", "卒見証", "在学証", "卒業証", "学籍証明"
    ],

    # 留学関連
    "留学": [
        "留学", "海外", "協定校", "IEC", "国際交流", "交換留学", "半期留学", "短期海外研修",
        "韓国", "中国", "台湾", "タイ", "ベトナム", "アメリカ", "イギリス", "フランス",
        "インドネシア", "マレーシア", "ルーマニア", "ホームステイ", "UC Davis",
        "エセックス大学", "チェンマイ大学", "TOEIC", "単位認定", "60単位",
        # 俗語・略語
        "トフル", "トーイック", "エルツ", "IELTS", "スコア足りない", "留学費用", "留学奨学金"
    ],

    # 学生支援
    "学生支援": [
        "サークル", "部活", "ボランティア", "障がい", "障害", "データサイエンス",
        "AI教育プログラム", "札幌圏大学単位互換制度", "国内留学制度", "沖縄国際大学",
        "関東学院大学", "コーディネーター", "修学支援", "情報保障", "PCテイク", "通学介助",
        "別室受験", "サポートセンター", "学生支援課",
        # 俗語・略語
        "部活費", "サークル費", "ボラ活", "支援制度", "障サポ"
    ],

    # 経済支援
    "経済支援": [
        "奨学金", "授業料", "学費", "免除", "支援金", "学費支援", "家計急変", "被災学生",
        "授業料減免",
        # 俗語・略語
        "奨学", "奨学金免除", "学費免除", "学費タダ", "授業料免除", "給付型", "貸与型"
    ],

    # 資格関連
    "資格": [
        "資格", "免許", "講座", "キャリア", "教員免許状", "学芸員", "Teaching License",
        "Museum Curator", "中学", "高校", "小学校", "特別支援学校", "人文学部", "法学部",
        "経済経営学部", "人間科学科", "英語英米文学科", "こども発達学科", "法律学科",
        "経済学科", "経営学科", "社会", "英語", "地歴", "公民", "商業", "教育実習",
        "教職実践演習", "日商簿記検定", "実用英語技能検定",
        # 俗語・略語
        "教免", "教採", "教職課程", "簿記2級", "英検2級", "英検準1", "TOEIC650"
    ],

    # 相談関連
    "相談": [
        "相談", "カウンセリング", "悩み", "メンタル", "ハラスメント", "トラブル",
        # 俗語・略語
        "メンヘラ", "メンタルやられた", "しんどい", "パワハラ", "セクハラ", "モラハラ"
    ],

    # 施設関連
    "施設": [
        "図書館", "購買", "場所", "どこ", "Wi-Fi", "PC", "パソコン", "教室", "体育館",
        "グラウンド", "駐車場", "スポーツ施設", "江別第2キャンパス", "メインアリーナ",
        "サブアリーナ", "上靴", "江別キャンパス", "新札幌キャンパス", "C館",
        # 俗語・略語
        "図書", "ラーニングコモンズ", "自習室", "食堂", "カフェ", "購買部", "ジム"
    ],

    # 生協関連
    "生協": [
        "生協", "コープ", "教科書", "共済", "組合員", "購買", "食堂", "書籍", "パソコン",
        "カフェテリア", "学内ショップ", "メニュー",
        # 俗語・略語
        "学食", "学食メニュー", "生協食堂", "生協カード", "生協ポイント"
    ],

    # 履修関連
    "履修": [
        "履修", "科目", "単位", "必修", "履修登録", "セメスター制度", "前期", "後期",
        "124単位", "卒業要件", "45時間", "事前学修", "事後学修", "履修登録単位数",
        "上限", "42単位", "48単位", "面接授業", "遠隔授業", "英語IA", "英語IB",
        "スポーツA", "スポーツB", "バドミントン", "バレーボール", "バスケットボール",
        "卓球", "教養科目", "専門"],
    # 履修関連
    "履修": [
        "履修", "科目", "単位", "必修", "履修登録", "セメスター制度", "前期", "後期",
        "124単位", "卒業要件", "45時間", "事前学修", "事後学修", "履修登録単位数",
        "上限", "42単位", "48単位", "面接授業", "遠隔授業", "英語IA", "英語IB",
        "スポーツA", "スポーツB", "バドミントン", "バレーボール", "バスケットボール",
        "卓球", "教養科目", "専門科目", "Moodle", "履修登録取消制度", "抽選科目",
        "他大学", "既修得単位",
        # 俗語・略語
        "履サボ", "履修落ち", "履修漏れ", "履修ミス", "履修取消", "履修キャンセル",
        "単位落ち", "単落ち", "単位足りない", "必修落ち", "必修サボ", "抽選落ち",
        "履修登録システム", "履修登録エラー"
    ],

    # 連絡・提出関連
    "連絡・提出": [
        "情報ポータル", "掲示", "掲示板", "メールアドレス", "提出物", "提出時間", "期限",
        "時刻", "レポート表紙", "学籍番号", "氏名", "時間割", "学科", "年", "ページ数",
        "教育支援課",
        # 俗語・略語
        "ポータル", "ポタ", "掲示見ろ", "メール確認", "レポ提出", "レポ出す", "課題提出",
        "締切", "デッドライン", "DL", "〆切", "遅延提出", "遅れ提出", "再提出"
    ]
}

# データベースから読み込むフォールバック情報を格納するグローバル変数
g_category_fallbacks: Dict[str, Dict[str, Any]] = {}

# --------------------------------------------------------------------------
# 3. 内部コンポーネントの定義
# --------------------------------------------------------------------------

def split_text(text: str, max_length: int = 1000, overlap: int = 100) -> List[str]:
    """テキストをチャンクに分割"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_length
        chunks.append(text[start:end])
        start += max_length - overlap
    return chunks

def format_urls_as_links(text: str) -> str:
    """URLをHTMLリンクに変換"""
    # 最初に単体のURLを変換
    url_pattern = r'(?<!\]\()(https?://[^\s\[\]()<>]+)(?!\))'
    text = re.sub(url_pattern, r'<a href="\1" target="_blank">\1</a>', text)
    
    # 次にマークダウン形式のリンク [テキスト](URL) を変換
    md_link_pattern = r'\[([^\]]+)\]\((https?://[^\s\)]+)\)'
    text = re.sub(md_link_pattern, r'<a href="\2" target="_blank">\1</a>', text)
    
    return text

# レート制限対応のヘルパー関数
async def safe_generate_content(model, prompt, stream=False, max_retries=3):
    """レート制限を考慮した安全なコンテンツ生成"""
    for attempt in range(max_retries):
        try:
            if stream:
                return await model.generate_content_async(
                    prompt, 
                    stream=True,
                    generation_config=GenerationConfig(
                        max_output_tokens=1024,
                        temperature=0.7
                    )
                )
            else:
                return await model.generate_content_async(
                    prompt,
                    generation_config=GenerationConfig(
                        max_output_tokens=512,
                        temperature=0.3
                    )
                )
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                if attempt < max_retries - 1:
                    # エラーメッセージから待機時間を抽出
                    wait_time = 15  # デフォルト15秒
                    if "retry in" in error_str:
                        try:
                            import re
                            match = re.search(r'retry in (\d+(?:\.\d+)?)s', error_str)
                            if match:
                                wait_time = float(match.group(1)) + 2
                        except:
                            pass
                    
                    logging.warning(f"API制限により{wait_time}秒待機中... (試行 {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise HTTPException(
                        status_code=429,
                        detail=f"APIクォータを超過しました。しばらく時間をおいてから再試行してください。"
                    )
            else:
                raise e
    
    raise HTTPException(status_code=500, detail="最大リトライ回数を超えました。")

class DocumentProcessor:
    """ドキュメント処理クラス"""
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
    """ウェブスクレイピングクラス"""
    def scrape(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            for element in soup(["script", "style", "header", "footer", "nav", "aside"]): 
                element.decompose()
            text = " ".join(t.strip() for t in soup.stripped_strings)
            return re.sub(r'\s+', ' ', text).strip()
        except Exception as e:
            logging.error(f"スクレイピングエラー ({url}): {e}")
            return ""

class LogManager:
    """ログ管理クラス（空実装版：ログを残さない）"""
    def __init__(self):
        # ファイル名は定義しておくが使わない
        self.logs_file = None
        self.feedback_file = None

    def generate_log_id(self) -> str:
        # 一応IDは返す（呼び出し側が依存している可能性があるため）
        return str(uuid.uuid4())

    def save_log(self, *args, **kwargs):
        # 何も保存しない
        pass

    def save_feedback(self, *args, **kwargs):
        # 何も保存しない
        pass

    def get_logs_with_feedback(self) -> list[dict]:
        # 常に空リストを返す
        return []

class SupabaseClientManager:
    """Supabaseクライアント管理クラス"""
    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)

    def get_db_type(self) -> str: 
        return "supabase"

    def insert_document(self, content: str, embedding: List[float], metadata: dict):
        """ドキュメントを挿入"""
        self.client.table("documents").insert({
            "content": content, 
            "embedding": embedding, 
            "metadata": metadata
        }).execute()

    def search_documents(self, collection_name: str, category: str, embedding: List[float], match_count: int) -> List[dict]:
        """ドキュメント検索"""
        params = {
            "p_collection_name": collection_name, 
            "p_category": category,
            "p_query_embedding": embedding, 
            "p_match_count": match_count
        }
        result = self.client.rpc("match_documents", params).execute()
        return result.data or []

    def get_documents_by_collection(self, collection_name: str) -> List[dict]:
        """コレクション内のドキュメント一覧を取得"""
        result = self.client.table("documents").select("id, metadata").eq("metadata->>collection_name", collection_name).execute()
        return result.data or []
    
    def count_chunks_in_collection(self, collection_name: str) -> int:
        """コレクション内のチャンク数をカウント"""
        result = self.client.table("documents").select("id", count='exact').eq("metadata->>collection_name", collection_name).execute()
        return result.count or 0

    def get_distinct_categories(self, collection_name: str) -> List[str]:
        """コレクション内のカテゴリ一覧を取得"""
        try:
            result = self.client.rpc("get_distinct_categories", {"p_collection_name": collection_name}).execute()
            categories = [item['category'] for item in (result.data or []) if item.get('category')]
            return categories if categories else ["その他"]
        except Exception as e:
            logging.error(f"RPC 'get_distinct_categories' の呼び出しエラー: {e}")
            return ["その他"]

class SettingsManager:
    """設定管理クラス"""
    def __init__(self):
        self.settings = {
            "model": "gemini-1.5-flash-latest", 
            "collection": ACTIVE_COLLECTION_NAME, 
            "embedding_model": "text-embedding-004", 
            "top_k": 5
        }
        self.websocket_connections: List[WebSocket] = []
        self.settings_file = os.path.join(BASE_DIR, "shared_settings.json")
        self.load_settings()

    def load_settings(self):
        """設定ファイルから読み込み"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    self.settings.update(json.load(f))
        except Exception as e:
            logging.error(f"設定ファイルの読み込みエラー: {e}")

    def save_settings(self):
        """設定ファイルに保存"""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"設定ファイルの保存エラー: {e}")

    async def update_settings(self, new_settings: Dict[str, Any]):
        """設定を更新してWebSocketでブロードキャスト"""
        self.settings.update(new_settings)
        self.save_settings()
        await self.broadcast_settings()

    async def add_websocket(self, websocket: WebSocket):
        """WebSocket接続を追加"""
        await websocket.accept()
        self.websocket_connections.append(websocket)

    def remove_websocket(self, websocket: WebSocket):
        """WebSocket接続を削除"""
        if websocket in self.websocket_connections:
            self.websocket_connections.remove(websocket)

    async def broadcast_settings(self):
        """設定をすべてのWebSocket接続にブロードキャスト"""
        message = {"type": "settings_update", "data": self.settings}
        disconnected = []
        
        for conn in self.websocket_connections:
            try:
                await conn.send_json(message)
            except:
                disconnected.append(conn)
        
        for conn in disconnected:
            self.remove_websocket(conn)

async def create_fallback_response_from_db(category: str, model: str) -> str:
    """
    DBに情報がない場合、カテゴリに応じたフォールバック応答をDBから取得して生成する。
    URLがあれば要約を試み、失敗すれば静的応答を返す。
    """
    info = g_category_fallbacks.get(category)
    
    # カテゴリに対応するフォールバック情報がDBにない場合
    if not info:
        return "申し訳ありませんが、お尋ねの件について情報が見つかりませんでした。[大学公式サイト](https://www.sgu.ac.jp/)をご確認ください。"

    # URL要約を試みる
    url_to_summarize = info.get("url_to_summarize")
    if url_to_summarize:
        try:
            web_scraper = WebScraper()
            content = web_scraper.scrape(url_to_summarize)
            if content and len(content) > 100:
                prompt = f"以下の大学公式サイトの内容を学生向けに分かりやすく要約してください：\n\n{content[:4000]}"
                gemini_model = genai.GenerativeModel(model)
                response = await safe_generate_content(gemini_model, prompt, stream=False)
                # response.text が None でないことを確認
                if response and response.text:
                    return f"**▼ {category}に関する公式情報**\n{response.text}\n\n詳細は[こちら]({url_to_summarize})をご確認ください。"
        except Exception as e:
            logging.warning(f"URL要約エラー ({url_to_summarize}): {e}")
    
    # URL要約が失敗したか、URLが元々ない場合は、静的応答を返す
    return info.get("static_response", "情報が見つかりませんでした。")


# --------------------------------------------------------------------------
# 4. FastAPIアプリケーションのセットアップ
# --------------------------------------------------------------------------
db_client: Optional[SupabaseClientManager] = None
settings_manager: Optional[SettingsManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理"""
    global db_client, settings_manager, g_category_fallbacks
    logging.info("--- アプリケーション起動処理開始 ---")
    
    # 設定マネージャー初期化
    settings_manager = SettingsManager()
    
    # Supabase初期化
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            db_client = SupabaseClientManager(url=SUPABASE_URL, key=SUPABASE_KEY)
            logging.info("Supabaseクライアントの初期化完了。")

            # DBからフォールバック情報を読み込む
            try:
                response = db_client.client.table("category_fallbacks").select("*").execute()
                if response.data:
                    for item in response.data:
                        g_category_fallbacks[item['category_name']] = {
                            "url_to_summarize": item.get('url_to_summarize'),
                            "static_response": item.get('static_response')
                        }
                    logging.info(f"{len(g_category_fallbacks)}件のカテゴリ別フォールバック情報をDBからロードしました。")
            except Exception as e:
                logging.error(f"DBからのフォールバック情報ロードに失敗: {e}")

        except Exception as e:
            logging.error(f"Supabase初期化エラー: {e}")
    else:
        logging.warning("Supabaseの環境変数が設定されていません。")
    
    yield
    logging.info("--- アプリケーション終了処理 ---")

app = FastAPI(lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=APP_SECRET_KEY)
from prometheus_fastapi_instrumentator import Instrumentator

# ... (FastAPIの他の設定) ...

# FastAPIアプリケーションのインスタンス化の後
# 例: app = FastAPI(lifespan=lifespan) の後

Instrumentator().instrument(app).expose(app)

# OAuth設定
oauth = OAuth()
if all([AUTH0_CLIENT_ID, AUTH0_CLIENT_SECRET, AUTH0_DOMAIN]):
    oauth.register(
        name='auth0', 
        client_id=AUTH0_CLIENT_ID, 
        client_secret=AUTH0_CLIENT_SECRET,
        server_metadata_url=f'https://{AUTH0_DOMAIN}/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid profile email'},
    )
else:
    logging.warning("Auth0の設定が不完全なため、認証機能は動作しません。")

# --- グローバルインスタンス ---
log_manager = LogManager()
document_processor = DocumentProcessor()
web_scraper = WebScraper()

# --- データモデル定義 ---
class ChatQuery(BaseModel):
    query: str
    model: str = "gemini-1.5-flash-latest"
    embedding_model: str = "text-embedding-004"
    top_k: int = 5
    collection: str = ACTIVE_COLLECTION_NAME

class ClientChatQuery(BaseModel):
    query: str

class ScrapeRequest(BaseModel):
    url: str
    category: str = "その他"
    embedding_model: str = "text-embedding-004"
    collection_name: str = ACTIVE_COLLECTION_NAME

class FeedbackRequest(BaseModel):
    log_id: str
    rating: str

class Settings(BaseModel):
    model: Optional[str] = None
    collection: Optional[str] = None
    embedding_model: Optional[str] = None
    top_k: Optional[int] = None

# WebSocket接続管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                self.disconnect(connection)

manager = ConnectionManager()

# --------------------------------------------------------------------------
# 5. APIエンドポイント定義
# --------------------------------------------------------------------------

# --- 認証関数 ---
def require_auth(request: Request):
    """認証が必要なエンドポイント用の依存関数"""
    user = request.session.get('user')
    if not user and all([AUTH0_CLIENT_ID, AUTH0_CLIENT_SECRET, AUTH0_DOMAIN]):
        raise HTTPException(status_code=307, headers={'Location': '/login'})
    return user

# --- 認証とHTML提供 ---
@app.get('/login')
async def login(request: Request):
    if 'auth0' not in oauth._clients: 
        raise HTTPException(status_code=500, detail="Auth0 is not configured.")
    return await oauth.auth0.authorize_redirect(request, request.url_for('auth'))

@app.get('/auth')
async def auth(request: Request):
    if 'auth0' not in oauth._clients: 
        raise HTTPException(status_code=500, detail="Auth0 is not configured.")
    token = await oauth.auth0.authorize_access_token(request)
    if userinfo := token.get('userinfo'): 
        request.session['user'] = dict(userinfo)
    return RedirectResponse(url='/admin')

@app.get('/logout')
async def logout(request: Request):
    request.session.pop('user', None)
    if not all([AUTH0_DOMAIN, AUTH0_CLIENT_ID]): 
        return RedirectResponse(url='/')
    return RedirectResponse(f"https://{AUTH0_DOMAIN}/v2/logout?returnTo={request.url_for('serve_client')}&client_id={AUTH0_CLIENT_ID}")

@app.get("/", response_class=FileResponse)
async def serve_client(): 
    return FileResponse(os.path.join(BASE_DIR, "client.html"))

@app.get("/admin", response_class=FileResponse)
async def serve_admin(request: Request):
    # 認証設定がある場合は認証をチェック
    if all([AUTH0_CLIENT_ID, AUTH0_CLIENT_SECRET, AUTH0_DOMAIN]):
        require_auth(request)
    return FileResponse(os.path.join(BASE_DIR, "admin.html"))

@app.get("/log", response_class=FileResponse)
async def serve_log(request: Request):
    # 認証設定がある場合は認証をチェック
    if all([AUTH0_CLIENT_ID, AUTH0_CLIENT_SECRET, AUTH0_DOMAIN]):
        require_auth(request)
    return FileResponse(os.path.join(BASE_DIR, "log.html"))

@app.get("/favicon.ico", include_in_schema=False)
async def favicon(): 
    return Response(status_code=204)

# --- ステータス確認 ---
@app.get("/health")
async def health_check(): 
    return {
        "status": "ok", 
        "database": db_client.get_db_type() if db_client else "uninitialized"
    }

@app.get("/gemini/status")
async def gemini_status():
    if not GEMINI_API_KEY: 
        return {"connected": False, "detail": "API key not configured"}
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        return {"connected": True, "models": models}
    except Exception as e: 
        return {"connected": False, "detail": str(e)}
    
@app.get("/healthz")
def health_check():
    return {"status": "ok"}

# --- コレクション管理API ---
@app.get("/collections")
async def get_collections():
    """コレクション一覧を返す（固定値）"""
    return [{"name": ACTIVE_COLLECTION_NAME, "count": db_client.count_chunks_in_collection(ACTIVE_COLLECTION_NAME) if db_client else 0}]

@app.post("/collections")
async def create_collection(request: dict):
    """コレクション作成（固定値のため何もしない）"""
    return {"message": f"コレクション「{ACTIVE_COLLECTION_NAME}」は既に存在しています"}

@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """コレクション削除（固定値のため削除不可）"""
    if collection_name == ACTIVE_COLLECTION_NAME:
        raise HTTPException(status_code=400, detail="このコレクションは削除できません")
    return {"message": "コレクションが見つかりません"}

# --- ナレッジ管理API ---
@app.get("/collections/{collection_name}/documents")
async def get_documents(collection_name: str):
    if not db_client: 
        raise HTTPException(503, "DB not initialized")
    return {
        "documents": db_client.get_documents_by_collection(collection_name),
        "count": db_client.count_chunks_in_collection(collection_name)
    }

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...), 
    embedding_model: str = Form("text-embedding-004"),
    collection_name: str = Form(ACTIVE_COLLECTION_NAME)
):
    if not db_client: 
        raise HTTPException(503, "DB not initialized")
    try:
        category = file.filename.split('_')[0] if '_' in file.filename else "その他"
        content = await file.read()
        chunks = document_processor.process(file.filename, content)
        
        for chunk in chunks:
            metadata = {
                "source": file.filename, 
                "collection_name": collection_name, 
                "category": category
            }
            # レート制限対応でembeddingを生成
            try:
                embedding_response = genai.embed_content(model=embedding_model, content=chunk)
                embedding = embedding_response["embedding"]
                db_client.insert_document(chunk, embedding, metadata)
                # API制限を考慮して少し待機
                await asyncio.sleep(1)
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    logging.warning("埋め込み生成でAPI制限に達しました。30秒待機します。")
                    await asyncio.sleep(30)
                    # 再試行
                    embedding_response = genai.embed_content(model=embedding_model, content=chunk)
                    embedding = embedding_response["embedding"]
                    db_client.insert_document(chunk, embedding, metadata)
                else:
                    raise
        
        return {"chunks": len(chunks)}
    except Exception as e: 
        logging.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scrape")
async def scrape_website(req: ScrapeRequest):
    if not db_client: 
        raise HTTPException(503, "DB not initialized")
    try:
        content = web_scraper.scrape(req.url)
        if not content: 
            raise HTTPException(status_code=400, detail="コンテンツを取得できませんでした。")
        
        chunks = split_text(content)
        for chunk in chunks:
            metadata = {
                "source": req.url, 
                "collection_name": req.collection_name, 
                "category": req.category
            }
            # レート制限対応
            try:
                embedding_response = genai.embed_content(model=req.embedding_model, content=chunk)
                embedding = embedding_response["embedding"]
                db_client.insert_document(chunk, embedding, metadata)
                await asyncio.sleep(1)
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    logging.warning("スクレイピングでAPI制限に達しました。30秒待機します。")
                    await asyncio.sleep(30)
                    embedding_response = genai.embed_content(model=req.embedding_model, content=chunk)
                    embedding = embedding_response["embedding"]
                    db_client.insert_document(chunk, embedding, metadata)
                else:
                    raise
        
        return {"chunks": len(chunks)}
    except Exception as e: 
        logging.error(f"Scrape error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- チャットAPI（RAG対応） ---
async def enhanced_chat_logic(request: Request, chat_req: ChatQuery):
    """統合チャットロジック（RAG対応）"""
    user_input = chat_req.query.strip()
    log_id = log_manager.generate_log_id()
    
    yield f"data: {json.dumps({'log_id': log_id})}\n\n"

    try:
        # --- 通常の学内情報FAQ処理 ---
        if not all([db_client, GEMINI_API_KEY]):
            error_msg = "システムが利用できません。管理者にお問い合わせください。"
            log_manager.save_log(log_id, user_input, error_msg, "", "エラー", False)
            yield f"data: {json.dumps({'content': error_msg})}\n\n"
            return

        # カテゴリ分類
        category = next((cat for cat, keys in KEYWORD_MAP.items() if any(key in user_input for key in keys)), "その他")
        
        full_response = ""
        context = ""
        has_specific_info = False

        try:
            # データベースから関連情報を検索
            query_embedding_response = genai.embed_content(model=chat_req.embedding_model, content=user_input)
            query_embedding = query_embedding_response["embedding"]
            
            search_results = db_client.search_documents(
                collection_name=chat_req.collection, 
                category=category,
                embedding=query_embedding, 
                match_count=chat_req.top_k
            )
            
            if search_results:
                context = "\n".join([doc['content'] for doc in search_results])
                has_specific_info = True
            
        except Exception as e:
            logging.error(f"データベース検索エラー: {e}")

        if has_specific_info:
            # データベースに情報がある場合
            prompt = f"""あなたは、札幌学院大学の学生を親切にサポートする、優秀なAIアシスタントです。

制約条件
以下の「参考情報」に記載されている内容だけを元に、学生からの「質問」に回答してください。質問が日本語の場合は日本語で、英語の場合は英語で回答してください。

「参考情報」に書かれていない事柄については、「ご質問の件について、参考情報の中には該当する情報が見つかりませんでした。」と正直に回答してください。あなた自身の知識で情報を補ったり、推測で回答したりすることは絶対に避けてください。

回答は、学生にとって分かりやすく、丁寧な言葉遣いを心がけてください。

出力形式
回答の冒頭には、質問の言語に応じて、以下のいずれかの一文を必ず入れてください。

日本語の場合: 「データベースの情報に基づき、ご質問にお答えします。」

英語の場合: "Based on the information provided, here is the answer to your question."

質問に対する答えを、要点をまとめて記述してください。

関連するURLが「参考情報」に含まれている場合は、質問の言語に応じて、以下のいずれかの見出しをつけ、箇条書きで分かりやすく記載してください。

日本語の場合: 「参考URL:」

英語の場合: "Reference URL(s):"

参考情報:
{context}

質問: {user_input}

回答:"""
            
            model = genai.GenerativeModel(chat_req.model)
            stream = await safe_generate_content(model, prompt, stream=True)
            
            # レスポンスを一旦すべて結合してからリンク変換
            temp_full_response = ""
            async for chunk in stream:
                if chunk.text:
                    temp_full_response += chunk.text
            
            full_response = format_urls_as_links(temp_full_response)
            yield f"data: {json.dumps({'content': full_response})}\n\n"
        
        else:
            # データベースに情報がない場合、フォールバック応答を生成
            fallback_response = await create_fallback_response_from_db(category, chat_req.model)
            # 送信する前にリンクを変換
            full_response = format_urls_as_links(fallback_response)
            yield f"data: {json.dumps({'content': full_response})}\n\n"
        
        # ログ保存 (full_responseはすでに変換済み)
        log_manager.save_log(log_id, user_input, full_response, context, category, has_specific_info)

    except Exception as e:
        error_message = f"エラーが発生しました: {str(e)}"
        logging.error(f"チャットロジックエラー: {e}\n{traceback.format_exc()}")
        log_manager.save_log(log_id, user_input, error_message, "", "エラー", False)
        yield f"data: {json.dumps({'content': error_message})}\n\n"

@app.post("/chat")
async def chat_endpoint(request: Request, query: ChatQuery):
    """管理者用チャットエンドポイント"""
    return StreamingResponse(enhanced_chat_logic(request, query), media_type="text/event-stream")

@app.post("/chat_for_client")
async def chat_for_client(request: Request, query: ClientChatQuery):
    """クライアント用チャットエンドポイント（固定設定）"""
    if not settings_manager:
        raise HTTPException(503, "設定マネージャーが初期化されていません")
    
    chat_query = ChatQuery(
        query=query.query,
        model=settings_manager.settings.get("model", "gemini-1.5-flash-latest"),
        embedding_model=settings_manager.settings.get("embedding_model", "text-embedding-004"),
        top_k=settings_manager.settings.get("top_k", 5),
        collection=settings_manager.settings.get("collection", ACTIVE_COLLECTION_NAME)
    )
    return StreamingResponse(enhanced_chat_logic(request, chat_query), media_type="text/event-stream")

# --- フィードバックAPI ---
@app.post("/feedback")
async def save_feedback(feedback: FeedbackRequest):
    """フィードバック保存"""
    try:
        log_manager.save_feedback(feedback.log_id, feedback.rating)
        return {"message": "フィードバックを保存しました"}
    except Exception as e:
        logging.error(f"フィードバック保存エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- ログ取得API ---
@app.get("/logs")
async def get_logs():
    """チャットログ取得"""
    try:
        logs = log_manager.get_logs_with_feedback()
        return logs
    except Exception as e:
        logging.error(f"ログ取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 設定管理API ---
@app.post("/settings")
async def update_settings(settings: Settings):
    """設定更新（WebSocket経由で通知）"""
    if not settings_manager:
        raise HTTPException(503, "設定マネージャーが初期化されていません")
    
    try:
        await settings_manager.update_settings(settings.dict(exclude_none=True))
        return {"message": "設定を更新しました"}
    except Exception as e:
        logging.error(f"設定更新エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- WebSocketエンドポイント ---
@app.websocket("/ws/settings")
async def websocket_endpoint(websocket: WebSocket):
    """設定同期用WebSocket"""
    if not settings_manager:
        await websocket.close(code=1011, reason="Settings manager not initialized")
        return
    
    await settings_manager.add_websocket(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # クライアントからのメッセージは特に処理しない
    except WebSocketDisconnect:
        settings_manager.remove_websocket(websocket)

# --------------------------------------------------------------------------
# 6. 開発用サーバー起動
# --------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

