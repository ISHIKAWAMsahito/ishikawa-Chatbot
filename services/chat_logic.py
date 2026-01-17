import logging
import uuid
import json
import asyncio
import re
import os
from typing import List, Dict, Any, AsyncGenerator, Optional
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
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
from services.utils import format_urls_as_links

# -----------------------------------------------------------------------------
# 1. 設定 & 定数定義
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

# AIメッセージテンプレート
AI_MESSAGES = {
    "SYSTEM_ERROR": "システムエラーが発生しました。管理者にお問い合わせください。",
    "NO_INFO": "申し訳ありません。その情報は見つかりませんでした。\n他に知りたいことはありますか？",
    "THINKING": "文献データベースを検索し、回答を作成しています..."
}

# -----------------------------------------------------------------------------
# 2. ヘルパー関数 (画像URL生成など)
# -----------------------------------------------------------------------------

def get_signed_image_url(image_path: str, expiry_seconds: int = 3600) -> Optional[str]:
    """
    Supabaseの非公開バケットから、一時的に有効な署名付きURLを発行する。
    
    Args:
        image_path (str): ストレージ内のファイルパス (例: 'a1b2c3d4.jpg')
        expiry_seconds (int): URLの有効期限（秒）。デフォルトは1時間。
        
    Returns:
        Optional[str]: 有効なURL。発行失敗時はNone。
    """
    if not image_path:
        return None
    
    try:
        # core.database経由でSupabaseクライアントにアクセス
        # バケット名は 'images' 固定 (images.pyの設定と一致させる)
        res = core_database.db_client.client.storage.from_("images").create_signed_url(image_path, expiry_seconds)
        
        # レスポンス形式の揺れに対応 (辞書型で返る場合とオブジェクトの場合があるため)
        if isinstance(res, dict):
            return res.get('signedURL')
        elif hasattr(res, 'signedURL'): # オブジェクトの場合
            return res.signedURL
        return None
    except Exception as e:
        logging.error(f"Failed to generate signed URL for {image_path}: {e}")
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
# 3. 参照元情報の構築ロジック (★重要修正箇所)
# -----------------------------------------------------------------------------

def _build_references(response_text: str, sources_map: Dict[int, Dict[str, Any]]) -> str:
    """
    回答で使用された情報源のリストをMarkdown形式で生成する。
    画像パスがある場合は、動的に署名付きURLを発行してリンクを追加する。
    
    Args:
        response_text: AIが生成した回答本文
        sources_map: {1: {"name": "文書名", "image_path": "path/to/img"}} 形式の辞書
    """
    unique_refs = []
    seen_sources = set()
    
    # 回答内で引用された番号 [1], [2] などを抽出
    cited_ids = set(map(int, re.findall(r'\[(\d+)\]', response_text)))
    
    # sources_map は順序付き辞書として扱う（Python 3.7+）
    for idx, data in sources_map.items():
        src_name = data.get('name', '不明な資料')
        image_path = data.get('image_path')
        
        # 重複排除用のキー
        source_key = f"{src_name}_{image_path}"
        if source_key in seen_sources:
            continue
            
        # 実際に引用されたか、もしくは上位2件（関連度が高い）なら表示
        if idx in cited_ids or idx <= 2:
            ref_line = f"* [{idx}] {src_name}"
            
            # ★ ここで画像をリンク化 ★
            if image_path:
                signed_url = get_signed_image_url(image_path)
                if signed_url:
                    # Markdownリンクとして追加
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
    session_id = query_obj.session_id or str(uuid.uuid4())
    user_message = query_obj.message
    feedback_id = str(uuid.uuid4())
    
    log_context(session_id, f"Start processing query: {user_message}")
    
    full_resp = ""
    history_manager = core_database.HistoryManager() # 履歴管理インスタンス

    try:
        # 1. 履歴の取得
        history = history_manager.get_recent(session_id, limit=PARAMS["MAX_HISTORY_LENGTH"])
        
        # 2. 文書検索 (Vector Search)
        # documentsテーブルから検索を実行
        search_results = core_database.query_vector_db(
            user_message, 
            match_threshold=0.3, 
            match_count=8
        )
        
        relevant_docs = search_results if search_results else []
        log_context(session_id, f"Found {len(relevant_docs)} relevant documents")

        # 3. コンテキスト情報の構築
        context_parts = []
        sources_map = {} # {id: {"name": str, "image_path": str}}

        for idx, doc in enumerate(relevant_docs, 1):
            # メタデータの安全な取得
            meta = doc.get('metadata', {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except:
                    meta = {}
            
            src = meta.get('source', '不明')
            title = meta.get('title', '')
            # images.py で保存した image_path を取得
            img_path = meta.get('image_path')
            
            display_name = f"{title} ({src})" if title else src
            
            # ★ 情報を辞書形式で保存（後で画像リンクを作るため）
            sources_map[idx] = {
                "name": display_name,
                "image_path": img_path 
            }
            
            # AIに渡すテキスト情報
            content_text = doc.get('content', '').replace('\n', ' ')
            context_parts.append(f"<doc id='{idx}' source='{src}'>\n{content_text}\n</doc>")

        context_str = "\n".join(context_parts)

        # 4. プロンプト作成
        system_instruction = f"""
あなたは大学の学生生活支援チャットボットです。
以下の「検索された学内規定・資料」に基づいて、学生の質問に親切かつ正確に答えてください。

### 制約事項
1. 回答は必ず提供された<doc>タグ内の情報に基づいて作成してください。
2. 文中の情報を使用する場合は、文末に必ず引用元番号 [1] の形式で示してください。
3. 複数の情報を組み合わせる場合も、それぞれの出典を [1][2] のように明記してください。
4. 提供された情報に答えがない場合は、正直に「申し訳ありません、その情報は見つかりませんでした」と答えてください。
5. 憶測で回答しないでください。

### 検索された学内規定・資料
{context_str}

### 会話履歴
{history}
"""
        
        # Geminiモデルの準備
        model = genai.GenerativeModel(
            model_name=USE_MODEL,
            generation_config=GenerationConfig(temperature=0.3),
            safety_settings=SAFETY_SETTINGS,
            system_instruction=system_instruction
        )
        
        chat = model.start_chat(history=[])
        
        # 5. ストリーミング回答生成
        response_stream = await chat.send_message_async(user_message, stream=True)
        
        async for chunk in response_stream:
            if chunk.text:
                text_chunk = chunk.text
                full_resp += text_chunk
                # フロントエンドへ送信
                yield send_sse({'content': text_chunk})
                await asyncio.sleep(0.01) # ストリーム安定化

        # 6. 参照リンクの生成と送信
        if "申し訳ありません" not in full_resp and "情報は見つかりませんでした" not in full_resp:
            # ここで画像リンク付きの参照リストを作成
            refs_text = _build_references(full_resp, sources_map)
            
            if refs_text:
                yield send_sse({'content': refs_text})
                full_resp += refs_text
        
        # 履歴への保存
        history_manager.add(session_id, "user", user_message)
        history_manager.add(session_id, "assistant", full_resp)

    except Exception as e:
        log_context(session_id, f"Pipeline Error: {e}", "error")
        # エラー時もユーザーには優しく返す
        if not full_resp:
            yield send_sse({'content': AI_MESSAGES["SYSTEM_ERROR"]})
            
    finally:
        # フィードバック要求トリガー
        yield send_sse({'show_feedback': True, 'feedback_id': feedback_id})

# -----------------------------------------------------------------------------
# 5. 分析機能 (変更なし)
# -----------------------------------------------------------------------------
async def analyze_feedback_trends(logs: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
    if not logs:
        yield send_sse({'content': '分析対象データがありません。'})
        return
    summary = "\n".join([f"- 評価:{l.get('rating','-')} | {l.get('comment','-')[:100]}" for l in logs[:50]])
    prompt = f"以下のログを分析してください:\n{summary}"
    try:
        model = genai.GenerativeModel(USE_MODEL)
        stream = await model.generate_content_async(prompt, stream=True)
        async for chunk in stream:
            if chunk.text:
                yield send_sse({'content': chunk.text})
    except Exception as e:
        yield send_sse({'content': f"分析エラー: {str(e)}"})