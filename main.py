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

from fastapi import Request
from fastapi.responses import FileResponse

from collections import defaultdict
from typing import Dict, List

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

# Auth0設定 (これは元のコードのもので、新しいJWT認証とは別)
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

SUPER_ADMIN_EMAILS_STR = os.getenv("SUPER_ADMIN_EMAILS", "")
SUPER_ADMIN_EMAILS = [email.strip() for email in SUPER_ADMIN_EMAILS_STR.split(',') if email.strip()]

ALLOWED_CLIENT_EMAILS_STR = os.getenv("ALLOWED_CLIENT_EMAILS", "")
ALLOWED_CLIENT_EMAILS = [email.strip() for email in ALLOWED_CLIENT_EMAILS_STR.split(',') if email.strip()]
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
        "特別欠席", "感染症", "教育実習", "クラブ", "学校保健安全法","課外活動","大会","出場"
        # 俗語・略語
        "サボる", "サボ", "バックれる", "体調不良", "風邪", "熱", "寝坊", "遅刻", "遅延証明","講義休みたい"
    ],
    # 試験関連
    "試験": [
        "試験", "テスト", "追試", "再試", "成績", "レポート", "課題", "定期試験", "60分",
        "成績評価", "秀", "優", "良", "可", "不可", "GPA", "Grade Point Average",
        "成績確認", "不正行為", "カンニング", "受験許可証",
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
        "エセックス大学", "チェンマイ大学", "TOEIC", "単位認定", "60単位","在留"
        # 俗語・略語
        "トフル", "トーイック", "エルツ", "IELTS", "スコア足りない", "留学費用", "留学奨学金"
    ],
    # 学生支援
    "学生支援": [
        "サークル", "部活", "ボランティア", "障がい", "障害", "データサイエンス",
        "AI教育プログラム", "札幌圏大学単位互換制度", "国内留学制度", "沖縄国際大学",
        "関東学院大学", "コーディネーター", "修学支援", "情報保障", "PCテイク", "通学介助",
        "別室受験", "サポートセンター", "学生支援課","窓口","事故","クラブ"
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
        "資格", "免許", "講座", "教員免許状", "学芸員", "Teaching License",
        "Museum Curator", "中学", "高校", "小学校", "特別支援学校", "人文学部", "法学部",
        "経済経営学部", "人間科学科", "英語英米文学科", "こども発達学科", "法律学科",
        "経済学科", "経営学科", "社会", "英語", "地歴", "公民", "商業", "教育実習",
        "教職実践演習", "日商簿記検定", "実用英語技能検定",
        # 俗語・略語
        "教免", "教採", "教職課程", "簿記2級", "英検2級", "英検準1", "TOEIC650"
    ],
    # 
    "相談": [
        "相談", "カウンセリング", "悩み", "メンタル", "ハラスメント", "トラブル",
        # 俗語・略語
        "メンヘラ", "メンタルやられた", "しんどい", "パワハラ", "セクハラ", "モラハラ"
    ],
    # 施設関連
    "施設": [
        "図書館", "場所", "どこ", "Wi-Fi", "PC", "パソコン", "教室", "体育館",
        "グラウンド", "駐車場", "スポーツ施設", "江別第2キャンパス", "メインアリーナ",
        "サブアリーナ", "上靴", "C館","施設","故障","破損","器物","F館","学生館","体育センター","SGUホール","50年記念館","用具","体育センター","陸上競技場","テニスコート","バスケットコート","バレーコート","フットサルコート","多目的グランド","野 球 場","室内練習場","多目的グランド","ランニングロード","弓 道 場","洋 弓 場","合宿","ラウンジ","下宿","指定施設","アパート","マンション","寮","男子寮","女子寮","トレーニングルーム","シャワーコーナー","保健室","保健センター","AED"
        # 俗語・略語
        "図書", "ラーニングコモンズ", "自習室", "食堂", "カフェ", "購買部", "ジム"
    ],
    # 生協関連
    "生協": [
        "生協", "コープ", "教科書", "共済", "組合員", "購買", "食堂", "書籍", "パソコン",
        "カフェテリア", "学内ショップ", "メニュー","自動車学校","教習所"
        # 俗語・略語
        "学食", "学食メニュー", "生協食堂", "生協カード", "生協ポイント","自学"
    ],
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
    ],

"キャリア支援": [
    "就職","キャリア","就活","面接"
],
"学則":[
    "学則","規定","規則","条項","条例","定め","基準","決まりごと"
],

"大学の概要":[
    "歴史","概要","沿革","開設","開学","募集停止","廃止",
]

}

# データベースから読み込むフォールバック情報を格納するグローバル変数
g_category_fallbacks: Dict[str, Dict[str, Any]] = {}
chat_histories: Dict[str, List[Dict[str, str]]] = defaultdict(list)
MAX_HISTORY_LENGTH = 20 
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
    url_pattern = r'(?<!\]\()(https?://[^\s\[\]()<>]+)(?!\))'
    text = re.sub(url_pattern, r'<a href="\1" target="_blank">\1</a>', text)
    md_link_pattern = r'\[([^\]]+)\]\((https?://[^\s\)]+)\)'
    text = re.sub(md_link_pattern, r'<a href="\2" target="_blank">\1</a>', text)
    return text

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
                    wait_time = 15
                    if "retry in" in error_str:
                        try:
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

class FeedbackManager:
    """フィードバック管理クラス（簡素化版）"""
    def __init__(self):
        self.feedback_file = os.path.join(BASE_DIR, "feedback.json")

    def save_feedback(self, feedback_id: str, rating: str, comment: str = ""):
        """フィードバックを保存"""
        try:
            feedback_data = []
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    feedback_data = json.load(f)
            feedback_entry = {
                "id": feedback_id,
                "rating": rating,
                "comment": comment,
                "timestamp": datetime.now(JST).isoformat()
            }
            feedback_data.append(feedback_entry)
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(feedback_data, f, ensure_ascii=False, indent=2)
            logging.info(f"フィードバック保存完了: {feedback_id} - {rating}")
        except Exception as e:
            logging.error(f"フィードバック保存エラー: {e}")
            raise

    def get_feedback_stats(self) -> Dict[str, Any]:
        """フィードバック統計を取得"""
        try:
            if not os.path.exists(self.feedback_file):
                return {"total": 0, "resolved": 0, "not_resolved": 0, "rate": 0}
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
            total = len(feedback_data)
            resolved = sum(1 for fb in feedback_data if fb['rating'] == 'resolved')
            not_resolved = total - resolved
            rate = (resolved / total * 100) if total > 0 else 0
            return {
                "total": total,
                "resolved": resolved,
                "not_resolved": not_resolved,
                "rate": round(rate, 1)
            }
        except Exception as e:
            logging.error(f"フィードバック統計取得エラー: {e}")
            return {"total": 0, "resolved": 0, "not_resolved": 0, "rate": 0}

class SupabaseClientManager:
    """Supabaseクライアント管理クラス"""
    def search_documents_by_vector(self, collection_name: str, embedding: List[float], match_count: int) -> List[dict]:
        """カテゴリで絞り込まずにベクトル検索を行う"""
        params = {
            "p_collection_name": collection_name,
            "p_query_embedding": embedding,
            "p_match_count": match_count
        }
        # 作成した新しいRPC関数 'match_documents_by_vector' を呼び出す
        result = self.client.rpc("match_documents_by_vector", params).execute()
        return result.data or []
    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)

    def get_db_type(self) -> str:
        return "supabase"

    def insert_document(self, content: str, embedding: List[float], metadata: dict):
        self.client.table("documents").insert({
            "content": content,
            "embedding": embedding,
            "metadata": metadata
        }).execute()

    def search_documents(self, collection_name: str, category: str, embedding: List[float], match_count: int) -> List[dict]:
        params = {
            "p_collection_name": collection_name,
            "p_category": category,
            "p_query_embedding": embedding,
            "p_match_count": match_count
        }
        result = self.client.rpc("match_documents", params).execute()
        return result.data or []

    def get_documents_by_collection(self, collection_name: str) -> List[dict]:
        result = self.client.table("documents").select("id, metadata").eq("metadata->>collection_name", collection_name).execute()
        return result.data or []

    def count_chunks_in_collection(self, collection_name: str) -> int:
        result = self.client.table("documents").select("id", count='exact').eq("metadata->>collection_name", collection_name).execute()
        return result.count or 0

    def get_distinct_categories(self, collection_name: str) -> List[str]:
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
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    self.settings.update(json.load(f))
        except Exception as e:
            logging.error(f"設定ファイルの読み込みエラー: {e}")

    def save_settings(self):
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"設定ファイルの保存エラー: {e}")

    async def update_settings(self, new_settings: Dict[str, Any]):
        self.settings.update(new_settings)
        self.save_settings()
        await self.broadcast_settings()

    async def add_websocket(self, websocket: WebSocket):
        await websocket.accept()
        self.websocket_connections.append(websocket)

    def remove_websocket(self, websocket: WebSocket):
        if websocket in self.websocket_connections:
            self.websocket_connections.remove(websocket)

    async def broadcast_settings(self):
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
    if not info:
        return "申し訳ありませんが、お尋ねの件について情報が見つかりませんでした。[大学公式サイト](https://www.sgu.ac.jp/)をご確認ください。"
    url_to_summarize = info.get("url_to_summarize")
    if url_to_summarize:
        try:
            web_scraper = WebScraper()
            content = web_scraper.scrape(url_to_summarize)
            if content and len(content) > 100:
                prompt = f"以下の大学公式サイトの内容を学生向けに分かりやすく要約してください：\n\n{content[:4000]}"
                gemini_model = genai.GenerativeModel(model)
                response = await safe_generate_content(gemini_model, prompt, stream=False)
                if response and response.text:
                    return f"**▼ {category}に関する公式情報**\n{response.text}\n\n詳細は[こちら]({url_to_summarize})をご確認ください。"
        except Exception as e:
            logging.warning(f"URL要約エラー ({url_to_summarize}): {e}")
    return info.get("static_response", "情報が見つかりませんでした。")


# --------------------------------------------------------------------------
# 4. FastAPIアプリケーションのセットアップ
# --------------------------------------------------------------------------
db_client: Optional[SupabaseClientManager] = None
settings_manager: Optional[SettingsManager] = None


# 8. lifespan関数を更新
@asynccontextmanager
async def lifespan(app: FastAPI):
    """認証システムを含むアプリケーションのライフサイクル管理"""
    global db_client, settings_manager, g_category_fallbacks
    logging.info("--- アプリケーション起動処理開始(認証システム対応) ---")

    settings_manager = SettingsManager()

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

    logging.info("--- アプリケーション終了処理(認証システム対応) ---")

app = FastAPI(lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=APP_SECRET_KEY)
from prometheus_fastapi_instrumentator import Instrumentator
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
    logging.warning("Auth0の設定が不完全なため、管理者ページの認証機能は動作しません。")

# --- グローバルインスタンス ---
feedback_manager = FeedbackManager()
document_processor = DocumentProcessor()
web_scraper = WebScraper()

# --- データモデル定義 ---
class ChatQuery(BaseModel):
    query: str
    model: str = "gemini-2.5-flash"
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
    feedback_id: str
    rating: str
    comment: str = ""

class AIResponseFeedbackRequest(BaseModel):
    user_question: str
    ai_response: str
    rating: str  # 'good' or 'bad'

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
@app.post("/feedback/ai-response")
async def save_ai_response_feedback(feedback: AIResponseFeedbackRequest):
    try:
        # feedback.comment にAI回答 or ユーザーコメントを格納
        # created_atはPython側でISOフォーマットのタイムスタンプを格納
        # ratingは good/bad など
        db_client.client.table("anonymous_comments").insert({
            "comment": f"Q: {feedback.user_question}\nA: {feedback.ai_response}",  # 質問・回答をまとめてテキスト化
            "created_at": datetime.now(JST).isoformat(),
            "rating": feedback.rating
        }).execute()
        return {"message": "AI回答へのフィードバックを保存しました"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# main.py のどこか（APIエンドポイントが定義されているエリア）に追加

@app.get("/config")
def get_config():
    """
    フロントエンドが必要とする公開可能な設定（環境変数）を返す
    """
    return {
        "supabase_url": os.getenv("SUPABASE_URL"),
        "supabase_anon_key": os.getenv("SUPABASE_ANON_KEY")
    }
    



    
# --- 認証関数 (Auth0用) ---
def require_auth(request: Request):
    """管理者用認証"""
    user = request.session.get('user')
    if not user:
        raise HTTPException(status_code=307, headers={'Location': '/login'})
    
    user_email = user.get('email', '')
    allowed_domain_staff = '@sgu.ac.jp'

    # 教職員ドメイン、または、スーパー管理者リストに含まれているかチェック
    if (user_email.endswith(allowed_domain_staff) or
            user_email in SUPER_ADMIN_EMAILS): # ← ★★★ こちらに変更 ★★★
        return user
    else:
        raise HTTPException(status_code=403, detail="管理者ページへのアクセス権がありません。")

def require_auth_client(request: Request):
    """クライアント用認証"""
    user = request.session.get('user')
    if not user:
        raise HTTPException(status_code=307, headers={'Location': '/login'})
    
    user_email = user.get('email', '')
    allowed_domain_student = '@e.sgu.ac.jp' # 学生のメールドメインを定義**
    
    # 学生ドメインと許可リストのみをチェック
    if (user_email.endswith(allowed_domain_student) or
            user_email in ALLOWED_CLIENT_EMAILS):
        return user
    else:
        raise HTTPException(status_code=403, detail="このサービスへのアクセスは許可されていません。")

# --- 認証とHTML提供 (Auth0用) ---
# main.pyの修正箇所（ルート定義部分のみ）
# 既存の重複したルート定義を削除し、以下に置き換えてください

# --- 認証とHTML提供 (Auth0用) ---
@app.get('/login')
async def login_auth0(request: Request):
    if 'auth0' not in oauth._clients:
        raise HTTPException(status_code=500, detail="Auth0 is not configured.")
    return await oauth.auth0.authorize_redirect(request, request.url_for('auth'))

@app.get('/auth')
async def auth(request: Request):
    """
    Auth0からのコールバックを処理し、ユーザー情報に基づいて
    適切なページ（/admin または /）にリダイレクトする。
    """
    if 'auth0' not in oauth._clients:
        raise HTTPException(status_code=500, detail="Auth0 is not configured.")
    
    try:
        token = await oauth.auth0.authorize_access_token(request)
    except Exception as e:
        logging.error(f"Auth0 access token error: {e}")
        return RedirectResponse(url='/login')

    if userinfo := token.get('userinfo'):
        # ユーザー情報をセッションに保存
        request.session['user'] = dict(userinfo)
        user_email = userinfo.get('email', '')
        
        # 定数を定義
        allowed_domain_staff = '@sgu.ac.jp'
        allowed_domain_student = '@e.sgu.ac.jp'

        # --- 権限に基づくリダイレクト判定 ---
        # 1. 管理者権限を持つか？ (教職員ドメイン または SUPER_ADMIN_EMAILSリスト)
        if (user_email.endswith(allowed_domain_staff) or
                user_email in SUPER_ADMIN_EMAILS):
            return RedirectResponse(url='/admin')

        # 2. クライアント（学生など）権限を持つか？ (学生ドメイン または ALLOWED_CLIENT_EMAILSリスト)
        elif (user_email.endswith(allowed_domain_student) or
                user_email in ALLOWED_CLIENT_EMAILS):
            return RedirectResponse(url='/')
        
        # 3. 上記のいずれにも該当しない場合は、許可されていないユーザー
        else:
            logging.warning(f"Unauthorized login attempt by: {user_email}")
            return RedirectResponse(url='/logout')
            
    # Auth0からuserinfoが取得できなかった場合
    logging.error("Failed to get userinfo from Auth0.")
    return RedirectResponse(url='/login')

@app.get('/logout')
async def logout(request: Request):
    request.session.pop('user', None)
    if not all([AUTH0_DOMAIN, AUTH0_CLIENT_ID]):
        return RedirectResponse(url='/')
    return RedirectResponse(f"https://{AUTH0_DOMAIN}/v2/logout?returnTo={request.url_for('serve_client')}&client_id={AUTH0_CLIENT_ID}")

# クライアント用ページ（学生用）
@app.get("/", response_class=FileResponse)
async def serve_client(request: Request, user: dict = Depends(require_auth_client)):
    return FileResponse(os.path.join(BASE_DIR, "client.html"))

# 管理者用メインページ
@app.get("/admin", response_class=FileResponse)
async def serve_admin(request: Request, user: dict = Depends(require_auth)):
    return FileResponse(os.path.join(BASE_DIR, "admin.html"))

# ★★★ 重要: DB管理ページ（管理者のみアクセス可能）★★★
# 重複を削除し、この1つだけ残します
@app.get("/DB.html", response_class=FileResponse)
async def serve_db_page(request: Request, user: dict = Depends(require_auth)):
    """DB管理ページ(DB.html)を提供するためのエンドポイント"""
    db_path = os.path.join(BASE_DIR, "DB.html")
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="DB.html not found")
    return FileResponse(db_path)

# CSS提供
@app.get("/style.css", response_class=FileResponse)
async def serve_css():
    """統一されたCSSファイル(style.css)を提供するためのエンドポイント"""
    return FileResponse(os.path.join(BASE_DIR, "style.css"))

# フィードバック統計ページ
@app.get("/feedback-stats", response_class=FileResponse)
async def serve_feedback_stats(request: Request, user: dict = Depends(require_auth)):
    return FileResponse(os.path.join(BASE_DIR, "feedback_stats.html"))

# Faviconエラー回避
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
def health_check_k8s():
    return {"status": "ok"}

# --- コレクション管理API ---
@app.get("/collections")
async def get_collections():
    return [{"name": ACTIVE_COLLECTION_NAME, "count": db_client.count_chunks_in_collection(ACTIVE_COLLECTION_NAME) if db_client else 0}]

@app.post("/collections")
async def create_collection(request: dict):
    return {"message": f"コレクション「{ACTIVE_COLLECTION_NAME}」は既に存在しています"}

@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    if collection_name == ACTIVE_COLLECTION_NAME:
        raise HTTPException(status_code=400, detail="このコレクションは削除できません")
    return {"message": "コレクションが見つかりません"}

# --- ナレッジ管理API ---
# --- ナレッジ管理API ---
@app.get("/collections/{collection_name}/documents")
async def get_documents(collection_name: str):
    if not db_client:
        raise HTTPException(503, "DB not initialized")
    return {
        "documents": db_client.get_documents_by_collection(collection_name),
        "count": db_client.count_chunks_in_collection(collection_name)
    }

# --- ドキュメント管理API（新規追加） ---
@app.get("/api/documents/all")
async def get_all_documents(user: dict = Depends(require_auth)):
    """全ドキュメントを取得（管理者のみ）"""
    if not db_client:
        raise HTTPException(503, "DB not initialized")
    try:
        result = db_client.client.table("documents").select("*").order("id", desc=True).limit(1000).execute()
        return {"documents": result.data or []}
    except Exception as e:
        logging.error(f"ドキュメント一覧取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/{doc_id}")
async def get_document_by_id(doc_id: int, user: dict = Depends(require_auth)):
    """特定のドキュメントを取得（管理者のみ）"""
    if not db_client:
        raise HTTPException(503, "DB not initialized")
    try:
        result = db_client.client.table("documents").select("*").eq("id", doc_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="ドキュメントが見つかりません")
        return result.data[0]
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"ドキュメント取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/documents/{doc_id}")
async def update_document(doc_id: int, request: Dict[str, Any], user: dict = Depends(require_auth)):
    """ドキュメントを更新（管理者のみ）。content更新時にベクトルも再生成する。"""
    if not db_client or not settings_manager:
        raise HTTPException(503, "DBまたは設定マネージャーが初期化されていません")
    try:
        update_data = {}

        # 1. contentが更新される場合は、ベクトルも再生成する
        if "content" in request:
            new_content = request["content"]
            update_data["content"] = new_content
            
            # 設定マネージャーから現在のエンベディングモデルを取得
            embedding_model = settings_manager.settings.get("embedding_model", "text-embedding-004")
            
            logging.info(f"ドキュメント {doc_id} のコンテンツが変更されたため、ベクトルを再生成します...")
            try:
                # 新しいコンテンツでベクトルを生成
                embedding_response = genai.embed_content(
                    model=embedding_model,
                    content=new_content
                )
                update_data["embedding"] = embedding_response["embedding"]
                logging.info(f"ドキュメント {doc_id} のベクトル再生成が完了しました。")
            except Exception as e:
                # APIのレート制限などを考慮
                if "429" in str(e) or "quota" in str(e).lower():
                    logging.warning("ベクトル再生成でAPI制限に達しました。30秒待機します。")
                    await asyncio.sleep(30)
                    embedding_response = genai.embed_content(
                        model=embedding_model,
                        content=new_content
                    )
                    update_data["embedding"] = embedding_response["embedding"]
                else:
                    logging.error(f"ベクトル再生成エラー: {e}")
                    raise HTTPException(status_code=500, detail=f"ベクトル再生成中にエラーが発生しました: {e}")

        # 2. メタデータは独立して更新可能
        if "metadata" in request:
            update_data["metadata"] = request["metadata"]

        if not update_data:
            raise HTTPException(status_code=400, detail="更新するデータがありません")
        
        result = db_client.client.table("documents").update(update_data).eq("id", doc_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="ドキュメントが見つかりません")
        
        logging.info(f"ドキュメント {doc_id} を更新しました（管理者: {user.get('email')}）")
        return {"message": "ドキュメントを更新しました", "document": result.data[0]}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"ドキュメント更新エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: int, user: dict = Depends(require_auth)):
    """ドキュメントを削除（管理者のみ）"""
    if not db_client:
        raise HTTPException(503, "DB not initialized")
    try:
        result = db_client.client.table("documents").delete().eq("id", doc_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="ドキュメントが見つかりません")
        
        logging.info(f"ドキュメント {doc_id} を削除しました（管理者: {user.get('email')}）")
        return {"message": "ドキュメントを削除しました"}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"ドキュメント削除エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(
     user: dict = Depends(require_auth),
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
            try:
                embedding_response = genai.embed_content(model=embedding_model, content=chunk)
                embedding = embedding_response["embedding"]
                db_client.insert_document(chunk, embedding, metadata)
                await asyncio.sleep(1)
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    logging.warning("埋め込み生成でAPI制限に達しました。30秒待機します。")
                    await asyncio.sleep(30)
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

# 686行目から734行目までの不完全なenhanced_chat_logic関数を削除

# その後、738行目付近に以下の関数を追加(グローバルスコープ)
def get_or_create_session_id(request: Request) -> str:
    """セッションIDを取得または新規作成"""
    session_id = request.session.get('chat_session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session['chat_session_id'] = session_id
    return session_id

def add_to_history(session_id: str, role: str, content: str):
    """チャット履歴に追加"""
    chat_histories[session_id].append({
        "role": role,
        "content": content
    })
    if len(chat_histories[session_id]) > MAX_HISTORY_LENGTH:
        chat_histories[session_id] = chat_histories[session_id][-MAX_HISTORY_LENGTH:]

def get_history(session_id: str) -> List[Dict[str, str]]:
    """チャット履歴を取得"""
    return chat_histories.get(session_id, [])

def clear_history(session_id: str):
    """チャット履歴をクリア"""
    if session_id in chat_histories:
        del chat_histories[session_id]

# 次に、enhanced_chat_logic関数を正しく定義
# main.py の916行目あたりから

async def enhanced_chat_logic(request: Request, chat_req: ChatQuery):
    user_input = chat_req.query.strip()
    feedback_id = str(uuid.uuid4())
    
    # セッションIDを取得
    session_id = get_or_create_session_id(request)
    
    yield f"data: {json.dumps({'feedback_id': feedback_id})}\n\n"
    
    try:
        if not all([db_client, GEMINI_API_KEY]):
            error_msg = "システムが利用できません。管理者にお問い合わせください。"
            yield f"data: {json.dumps({'content': error_msg})}\n\n"
            return

        context = ""
        has_specific_info = False
        MIN_SIMILARITY_THRESHOLD = 0.75 # 類似度のしきい値

        # --- メインの検索ロジック：ベクトル検索を優先 ---
        try:
            query_embedding_response = genai.embed_content(model=chat_req.embedding_model, content=user_input)
            query_embedding = query_embedding_response["embedding"]
            
            # ▼▼ インデントを修正した箇所 ▼▼
            # カテゴリで絞らない専用のベクトル検索メソッドを呼び出す
            search_results = db_client.search_documents_by_vector(
                collection_name=chat_req.collection,
                embedding=query_embedding,
                match_count=chat_req.top_k
            )

            # 類似度がしきい値以上の結果のみをコンテキストに追加
            if search_results and search_results[0]['similarity'] >= MIN_SIMILARITY_THRESHOLD:
                # contentが存在する結果のみをフィルタリングして結合
                filtered_content = [doc['content'] for doc in search_results if doc.get('content')]
                context = "\n".join(filtered_content)
                has_specific_info = True

        except Exception as e:
            logging.error(f"データベース検索エラー: {e}")

        # 履歴を取得してコンテキストに追加
        history = get_history(session_id)
        history_context = ""
        if history:
            history_context = "\n\n過去の会話履歴:\n"
            for msg in history[-6:]:
                role_label = "学生" if msg["role"] == "user" else "AI"
                history_context += f"{role_label}: {msg['content']}\n"

        if has_specific_info:
            # プロンプトはご自身のものを使用してください
            prompt = f"""あなたは、札幌学院大学の学生を親切にサポートする、優秀なAIアシスタントです。

【重要な制約】
1. 以下の「参考情報」と「過去の会話履歴」に記載されている内容のみを使って回答してください。
2. あなた自身の一般知識や推測で情報を補ったり、他大学の事例を参考にすることは絶対に禁止です。
3. 情報が不足している場合は、「ご質問の件について、データベースには該当する情報が見つかりませんでした。詳細は教育支援課・学生支援課などの窓口へお問い合わせください。」と正直に回答してください。
4. 質問が日本語の場合は日本語で、英語の場合は英語で回答してください。
5. 回答は、学生にとって分かりやすく、丁寧な言葉遣いを心がけてください。

過去の会話履歴:
{history_context}

参考情報:
{context}

出力形式
- 回答の冒頭には「データベースの情報に基づき、ご質問にお答えします。」(英語の場合: "Based on the information provided, here is the answer to your question.")
- 質問に対する答えを、要点をまとめて記述してください。
- 関連するURLがある場合は「参考URL:」として箇条書きで記載してください。

質問: {user_input}

回答:"""
            model = genai.GenerativeModel(chat_req.model)
            stream = await safe_generate_content(model, prompt, stream=True)
            
            temp_full_response = ""
            try:
                async for chunk in stream:
                    if chunk.text:
                        temp_full_response += chunk.text
            except StopAsyncIteration:
                logging.warning("APIから空のストリームが返されました。セーフティ設定によるブロックの可能性があります。")
                temp_full_response = "申し訳ありませんが、そのご質問にはお答えできません。質問の内容を変えて、もう一度お試しください。"

            if not temp_full_response.strip():
                 temp_full_response = "回答を生成できませんでした。もう一度お試しください。"

            full_response = format_urls_as_links(temp_full_response)
            
            add_to_history(session_id, "user", user_input)
            add_to_history(session_id, "assistant", temp_full_response)
            
            yield f"data: {json.dumps({'content': full_response})}\n\n"
        else:
            # --- フォールバック処理：ベクトル検索で情報が見つからなかった場合 ---
            # ここではじめて KEYWORD_MAP を使い、カテゴリを判定する
            category = next((cat for cat, keys in KEYWORD_MAP.items() if any(key in user_input for key in keys)), "その他")
            
            fallback_response = await create_fallback_response_from_db(category, chat_req.model)
            full_response = format_urls_as_links(fallback_response)
            
            add_to_history(session_id, "user", user_input)
            add_to_history(session_id, "assistant", fallback_response)
            
            yield f"data: {json.dumps({'content': full_response})}\n\n"

        yield f"data: {json.dumps({'show_feedback': True, 'feedback_id': feedback_id})}\n\n"
    except Exception as e:
        error_message = f"エラーが発生しました: {str(e)}"
        logging.error(f"チャットロジックエラー: {e}\n{traceback.format_exc()}")
        yield f"data: {json.dumps({'content': error_message})}\n\n"

# --- ここまでが enhanced_chat_logic 関数 ---

# 履歴管理用エンドポイント
@app.get("/chat/history")
async def get_chat_history(request: Request, user: dict = Depends(require_auth_client)):
    """現在のセッションのチャット履歴を取得"""
    session_id = get_or_create_session_id(request)
    history = get_history(session_id)
    return {"history": history}

@app.delete("/chat/history")
async def delete_chat_history(request: Request, user: dict = Depends(require_auth_client)):
    """現在のセッションのチャット履歴を削除"""
    session_id = get_or_create_session_id(request)
    clear_history(session_id)
    return {"message": "履歴をクリアしました"}

@app.post("/chat")
async def chat_endpoint(request: Request, query: ChatQuery):
    return StreamingResponse(enhanced_chat_logic(request, query), media_type="text/event-stream")

# 7. 既存のチャットエンドポイントを認証必須に変更
@app.post("/chat_for_client")
async def chat_for_client_auth(request: Request, query: ClientChatQuery, user: dict = Depends(require_auth_client)):
    """認証されたクライアント用チャットエンドポイント"""
    if not settings_manager:
        raise HTTPException(503, "Settings manager not initialized")

    logging.info(f"Chat request from user: {user.get('email', 'N/A')}")

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
    try:
        feedback_manager.save_feedback(feedback.feedback_id, feedback.rating, feedback.comment)
        return {"message": "フィードバックを保存しました"}
    except Exception as e:
        logging.error(f"フィードバック保存エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feedback/stats")
async def get_feedback_stats():
    try:
        stats = feedback_manager.get_feedback_stats()
        return stats
    except Exception as e:
        logging.error(f"フィードバック統計取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 設定管理API ---
@app.post("/settings")
async def update_settings(settings: Settings):
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
    if not settings_manager:
        await websocket.close(code=1011, reason="Settings manager not initialized")
        return
    await settings_manager.add_websocket(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        settings_manager.remove_websocket(websocket)

# --------------------------------------------------------------------------
# 6. 開発用サーバー起動
# --------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)