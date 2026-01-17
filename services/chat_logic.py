import logging
import uuid
import json
import asyncio
import re
import os
from typing import List, Dict, Any, AsyncGenerator, Optional
from concurrent.futures import ThreadPoolExecutor
import typing_extensions as typing

# 外部ライブラリ
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig
from fastapi import Request
from dotenv import load_dotenv

# 内部モジュール
from core.config import GEMINI_API_KEY
from core import database as core_database
from models.schemas import ChatQuery

# -----------------------------------------------------------------------------
# 1. 設定 & クラス定義
# -----------------------------------------------------------------------------
load_dotenv()
genai.configure(api_key=GEMINI_API_KEY)

# 使用モデル
USE_MODEL = "gemini-2.5-flash" 

# パラメータ
PARAMS = {
    "QA_SIMILARITY_THRESHOLD": 0.92, 
    "RERANK_SCORE_THRESHOLD": 5.5,   
    "MAX_HISTORY_LENGTH": 10,        
}

# セーフティ設定
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}

# AIメッセージ
AI_MESSAGES = {
    "SYSTEM_ERROR": "システムエラーが発生しました。管理者にお問い合わせください。",
    "NO_INFO": "申し訳ありません。その情報は見つかりませんでした。\n他に知りたいことはありますか？",
    "THINKING": "文献データベースを検索し、回答を作成しています..."
}

# --- ★HistoryManagerの修正（遅延初期化の実装）★ ---
class HistoryManager:
    def __init__(self):
        # インポート時のクラッシュを防ぐため、ここではクライアントを取得しません。
        pass

    @property
    def supabase(self):
        """
        実際に必要になったタイミングで最新のクライアントインスタンスを取得します。
        これにより、アプリ起動時の初期化順序による NoneType エラーを回避します。
        """
        # db_client 自体が None の場合、または client 属性がない場合のガード
        if core_database.db_client is None or getattr(core_database.db_client, 'client', None) is None:
            logging.error("Database client is not initialized yet.")
            # 呼び出し元でハンドリングできるよう、ここでは空の操作を許容するかエラーを投げる
            # ここでは安全に例外を投げ、呼び出し元の try-except でキャッチさせます
            raise RuntimeError("Database client is not ready")
        
        return core_database.db_client.client

    def get_recent(self, session_id: str, limit: int = 10) -> str:
        """指定したセッションの直近の会話履歴を取得し、テキスト形式で返す"""
        try:
            # self.supabase プロパティ経由でアクセス
            res = self.supabase.table("chat_history")\
                .select("role, content, created_at")\
                .eq("session_id", session_id)\
                .order("created_at", desc=True)\
                .limit(limit)\
                .execute()
            
            if not res.data:
                return ""
            
            # 古い順に並べ替え
            history_data = sorted(res.data, key=lambda x: x['created_at'])
            
            formatted_history = []
            for msg in history_data:
                role = "User" if msg['role'] == "user" else "Model"
                formatted_history.append(f"{role}: {msg['content']}")
            
            return "\n".join(formatted_history)
        except Exception as e:
            # クライアント未初期化エラーも含めてログ出力
            logging.error(f"Failed to fetch history: {e}")
            return ""

    def add(self, session_id: str, role: str, content: str):
        """会話履歴をDBに保存する"""
        try:
            # self.supabase プロパティ経由でアクセス
            self.supabase.table("chat_history").insert({
                "session_id": session_id,
                "role": role,
                "content": content
            }).execute()
        except Exception as e:
            logging.error(f"Failed to add history: {e}")

# インスタンス化 (これを api/chat.py からインポートできるようにする)
history_manager = HistoryManager()

# -----------------------------------------------------------------------------
# 2. ヘルパー関数
# -----------------------------------------------------------------------------

def get_or_create_session_id(session_id: Optional[str] = None) -> str:
    """セッションIDが存在しない場合に新規作成する"""
    return session_id if session_id else str(uuid.uuid4())

def get_signed_image_url(image_path: str, expiry_seconds: int = 3600) -> Optional[str]:
    """非公開バケットから署名付きURLを発行する"""
    if not image_path:
        return None
    try:
        # core.database 経由で Supabase クライアントにアクセス
        # 念のためここもガードを入れるのが安全ですが、既存ロジックを踏襲しつつエラーハンドリング内で処理
        if core_database.db_client is None or not hasattr(core_database.db_client, 'client'):
             logging.error("Database client not ready for signing URL")
             return None

        res = core_database.db_client.client.storage.from_("images").create_signed_url(image_path, expiry_seconds)
        
        # レスポンス形式の揺れに対応
        if isinstance(res, dict):
            return res.get('signedURL')
        elif hasattr(res, 'signedURL'):
            return res.signedURL
        return None
    except Exception as e:
        logging.error(f"Failed to generate signed URL: {e}")
        return None

def send_sse(data: dict) -> str:
    """SSE形式のデータを生成"""
    return json.dumps(data, ensure_ascii=False) + "\n\n"

def log_context(session_id: str, message: str, level: str = "info"):
    """ログ出力用ヘルパー"""
    log_msg = f"[Session: {session_id}] {message}"
    if level == "error":
        logging.error(log_msg)
    else:
        logging.info(log_msg)

# -----------------------------------------------------------------------------
# 3. 参照元情報の構築ロジック
# -----------------------------------------------------------------------------

def _build_references(response_text: str, sources_map: Dict[int, Dict[str, Any]]) -> str:
    """回答で使用された情報源のリストを生成。画像リンク付き。"""
    unique_refs = []
    seen_sources = set()
    cited_ids = set(map(int, re.findall(r'\[(\d+)\]', response_text)))
    
    for idx, data in sources_map.items():
        src_name = data.get('name', '不明な資料')
        image_path = data.get('image_path')
        
        source_key = f"{src_name}_{image_path}"
        if source_key in seen_sources:
            continue
            
        if idx in cited_ids or idx <= 2:
            ref_line = f"* [{idx}] {src_name}"
            
            # 画像があれば署名付きURLを発行してMarkdownリンクを追加
            if image_path:
                signed_url = get_signed_image_url(image_path)
                if signed_url:
                    ref_line += f" ([図表・資料を表示]({signed_url}))"
            
            unique_refs.append(ref_line)
            seen_sources.add(source_key)
            
    if unique_refs:
        return "\n\n### 参照元データ\n" + "\n".join(unique_refs)
    return ""

# -----------------------------------------------------------------------------
# 4. メインチャットロジック
# -----------------------------------------------------------------------------

async def enhanced_chat_logic(query_obj: ChatQuery, request: Request) -> AsyncGenerator[str, None]:
    session_id = get_or_create_session_id(query_obj.session_id)
    user_message = query_obj.message
    feedback_id = str(uuid.uuid4())
    
    log_context(session_id, f"Process start: {user_message}")
    
    full_resp = ""

    try:
        # 1. 履歴の取得
        history = history_manager.get_recent(session_id, limit=PARAMS["MAX_HISTORY_LENGTH"])
        
        # 2. 文書検索 (Vector Search)
        search_results = core_database.query_vector_db(
            user_message, 
            match_threshold=0.3, 
            match_count=8
        )
        relevant_docs = search_results if search_results else []

        # 3. コンテキスト情報の構築
        context_parts = []
        sources_map = {} # {id: {"name": str, "image_path": str}}

        for idx, doc in enumerate(relevant_docs, 1):
            meta = doc.get('metadata', {})
            # メタデータが文字列で保存されている場合のパース処理
            if isinstance(meta, str):
                try: meta = json.loads(meta)
                except: meta = {}
            
            src = meta.get('source', '不明')
            title = meta.get('title', '')
            img_path = meta.get('image_path')
            
            display_name = f"{title} ({src})" if title else src
            sources_map[idx] = {"name": display_name, "image_path": img_path}
            
            content_text = doc.get('content', '').replace('\n', ' ')
            context_parts.append(f"<doc id='{idx}' source='{src}'>\n{content_text}\n</doc>")

        context_str = "\n".join(context_parts)

        # 4. プロンプト作成
        system_instruction = f"""あなたは大学の学生生活支援チャットボットです。
以下の資料に基づいて答えてください。引用元番号 [1] を必ず付けてください。
資料にないことは「わかりません」と答え、憶測は避けてください。

### 検索された資料
{context_str}

### 会話履歴
{history}
"""
        model = genai.GenerativeModel(
            model_name=USE_MODEL,
            generation_config=GenerationConfig(temperature=0.3),
            safety_settings=SAFETY_SETTINGS,
            system_instruction=system_instruction
        )
        
        chat = model.start_chat(history=[])
        
        # 5. 回答生成 (ストリーミング)
        response_stream = await chat.send_message_async(user_message, stream=True)
        
        async for chunk in response_stream:
            if chunk.text:
                full_resp += chunk.text
                yield send_sse({'content': chunk.text})
                await asyncio.sleep(0.01)

        # 6. 参照リンクの付与
        if "わかりません" not in full_resp and "見つかりませんでした" not in full_resp:
            refs_text = _build_references(full_resp, sources_map)
            if refs_text:
                yield send_sse({'content': refs_text})
                full_resp += refs_text
        
        # 履歴保存
        history_manager.add(session_id, "user", user_message)
        history_manager.add(session_id, "assistant", full_resp)

    except Exception as e:
        log_context(session_id, f"Error: {e}", "error")
        # エラー時もユーザーにはメッセージを返す
        if not full_resp:
            yield send_sse({'content': AI_MESSAGES["SYSTEM_ERROR"]})
            
    finally:
        yield send_sse({'show_feedback': True, 'feedback_id': feedback_id})

# -----------------------------------------------------------------------------
# 5. 分析機能
# -----------------------------------------------------------------------------
async def analyze_feedback_trends(logs: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
    if not logs:
        yield send_sse({'content': 'データなし'})
        return
    summary = "\n".join([f"- {l.get('rating','-')} | {l.get('comment','-')[:50]}" for l in logs[:20]])
    try:
        model = genai.GenerativeModel(USE_MODEL)
        stream = await model.generate_content_async(f"分析してください:\n{summary}", stream=True)
        async for chunk in stream:
            if chunk.text: yield send_sse({'content': chunk.text})
    except Exception as e:
        yield send_sse({'content': str(e)})