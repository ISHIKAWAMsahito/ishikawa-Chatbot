import logging
import uuid
import json
import asyncio
import re
import random
from collections import defaultdict
from typing import List, Dict, Any, AsyncGenerator
from fastapi import Request, HTTPException
import google.generativeai as genai
from google.generativeai.types import (
    GenerationConfig, 
    HarmCategory, 
    HarmBlockThreshold
)

# -----------------------------------------------
# 外部モジュール・設定のインポート
# -----------------------------------------------
from core.config import GEMINI_API_KEY
from core import database as core_database
from models.schemas import ChatQuery
from services.utils import format_urls_as_links

# -----------------------------------------------
# アプリケーション設定値
# -----------------------------------------------
# RAG (Stage 1) の類似度閾値
STRICT_THRESHOLD = 0.85
RELATED_THRESHOLD = 0.80

# フォールバック (Stage 2) の類似度閾値
FALLBACK_SIMILARITY_THRESHOLD = 0.91

# RAGコンテキストの最大文字数 (トークン制限超過を避けるための簡易的な制限)
MAX_CONTEXT_CHAR_LENGTH = 15000

# 履歴の最大保持数 (永続化の際に利用)
MAX_HISTORY_LENGTH = 20

# AIが「見つからない」と判断したときのマジックストリング
AI_NOT_FOUND_MESSAGE = "ご質問いただいた内容については、関連する情報が見つかりませんでした。お手数ですが、大学の公式サイトをご確認いただくか、窓口までお問い合わせください。"

# -----------------------------------------------
# チャット履歴管理
# -----------------------------------------------
class ChatHistoryManager:
    """
    チャット履歴を管理するクラス。
    """
    def __init__(self):
        self._histories: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """指定されたセッションIDの履歴を取得"""
        return self._histories.get(session_id, [])

    def add_to_history(self, session_id: str, role: str, content: str):
        """履歴に新しいメッセージを追加"""
        # 試作品段階のため、履歴保存を一時的に停止
        return
        
        history = self._histories.get(session_id, [])
        history.append({"role": role, "content": content})
        if len(history) > MAX_HISTORY_LENGTH:
            history = history[-MAX_HISTORY_LENGTH:]
        self._histories[session_id] = history

    def clear_history(self, session_id: str):
        """指定されたセッションIDの履歴を削除"""
        if session_id in self._histories:
            del self._histories[session_id]

# マネージャーのインスタンスを作成
history_manager = ChatHistoryManager()

# -----------------------------------------------
# セッション管理
# -----------------------------------------------
def get_or_create_session_id(request: Request) -> str:
    """リクエストからセッションIDを取得、または新規作成"""
    session_id = request.session.get('chat_session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session['chat_session_id'] = session_id
    return session_id

# -----------------------------------------------
# API呼び出し (レート制限対応)
# -----------------------------------------------
async def safe_generate_content(model, prompt, stream=False, max_retries=3):
    """
    レート制限(429)を考慮した安全なコンテンツ生成。
    """
    for attempt in range(max_retries):
        try:
            config = GenerationConfig(
                max_output_tokens=4096 if stream else 512,
                temperature=0.1 if stream else 0.3
            )
            if stream:
                return await model.generate_content_async(prompt, stream=True, generation_config=config)
            else:
                return await model.generate_content_async(prompt, generation_config=config)

        except StopAsyncIteration:
            # これはセーフティフィルターが作動したときに発生する
            logging.error(f"APIが空のストリームを返しました (StopAsyncIteration)。セーフティフィルターが作動した可能性があります。")
            raise Exception("APIが空の応答を返しました。セーフティフィルターが作動した可能性があります。")

        except Exception as e:
            error_str = str(e)
            if ("429" in error_str or "quota" in error_str.lower()) and attempt < max_retries - 1:
                wait_time = 0
                try:
                    match = re.search(r'retry in (\d+(?:\.\d+)?)s', error_str)
                    if match:
                        wait_time = float(match.group(1)) + random.uniform(1, 3) 
                except Exception:
                    pass 
                if wait_time == 0:
                    wait_time = (2 ** attempt) * 5 + random.uniform(0, 1)
                logging.warning(f"API制限により {wait_time:.1f} 秒待機中... (試行 {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait_time)
                continue
            else:
                logging.error(f"API生成エラー (リトライ {attempt}): {e}", exc_info=True)
                raise e
                
    raise HTTPException(status_code=500, detail="APIの呼び出しに失敗しました（最大リトライ回数超過）。")


# -----------------------------------------------
# Stage 2 (Q&Aフォールバック) ロジック
# -----------------------------------------------
async def _run_stage2_fallback(
    query_embedding: List[float], 
    session_id: str, 
    user_input: str,
    feedback_id: str
) -> AsyncGenerator[str, None]:
    """
    Stage 2 (Q&Aベクトル検索) を実行し、結果をストリーミングする
    """
    logging.info(f"Stage 2 (Q&Aベクトル検索) を実行します。") # INFOのまま (Stage 2のログは通常不要なため)
    fallback_response = ""

    try:
        # クエリが空（ベクトル化失敗）の場合は検索しない
        if not query_embedding:
            raise ValueError("Q&A検索のためのクエリベクトルがありません。")

        fallback_results = core_database.db_client.search_fallback_qa(
            embedding=query_embedding,
            match_count=1
        )

        if fallback_results:
            best_match = fallback_results[0]
            best_sim = best_match.get('similarity', 0)
            # ▼▼▼ [ここから修正] ▼▼▼
            best_content_preview = best_match.get('content', 'N/A')[:100].replace('\n', ' ') + "..."

            # ログに「何を」マッチしたか出力する
            logging.info(f"--- Stage 2 検索結果 (Top 1) ---")
            logging.info(f"  [Sim: {best_sim:.4f}] Content: '{best_content_preview}'")

            if best_sim >= FALLBACK_SIMILARITY_THRESHOLD:
                logging.info(f"  -> [使用] 閾値 {FALLBACK_SIMILARITY_THRESHOLD} を超えたため、この回答を使用します。")
            # ▲▲▲ [ここまで修正] ▲▲▲
                fallback_response = f"""データベースに直接の情報は見つかりませんでしたが、関連する「よくあるご質問」がありましたのでご案内します。

---
{best_match['content']}
"""
            else:
                # ▼▼▼ [ここから修正] ▼▼▼
                logging.info(f"  -> [不使用] 閾値 {FALLBACK_SIMILARITY_THRESHOLD} 未満のため、この回答は使用しません。")
                # ▲▲▲ [ここまで修正] ▲▲▲
                fallback_response = "申し訳ありませんが、ご質問に関連する情報が見つかりませんでした。大学公式サイトをご確認いただくか、大学の窓口までお問い合わせください。"
        else:
            logging.info("Stage 2 RAG 失敗。Q&Aデータベースが空か、検索エラーです。")
            fallback_response = "申し訳ありませんが、ご質問に関連する情報が見つかりませんでした。大学公式サイトをご確認いただくか、大学の窓口までお問い合わせください。"

    except Exception as e_fallback:
        logging.error(f"Stage 2 (Q&A検索) でエラーが発生: {e_fallback}", exc_info=True)
        fallback_response = "申し訳ありません。現在、関連情報の検索中にエラーが発生しました。時間をおいて再度お試しください。"

    full_response = format_urls_as_links(fallback_response)
    
    # 履歴に追加
    history_manager.add_to_history(session_id, "user", user_input)
    history_manager.add_to_history(session_id, "assistant", full_response)
    
    # フォールバックの回答を送信
    yield f"data: {json.dumps({'content': full_response})}\n\n"
    
    # 5. 最終処理 (フィードバック表示トリガー)
    yield f"data: {json.dumps({'show_feedback': True, 'feedback_id': feedback_id})}\n\n"


# -----------------------------------------------
# メインのチャットロジック
# -----------------------------------------------
async def enhanced_chat_logic(request: Request, chat_req: ChatQuery):
    """
    RAG + Q&Aフォールバック対応のチャット処理 (ストリーミング)
    """
    user_input = chat_req.query.strip()
    feedback_id = str(uuid.uuid4())
    session_id = get_or_create_session_id(request)
    query_embedding = [] # 埋め込みベクトルを後で使えるようにスコープを上げる

    # 1. フィードバックIDをクライアントに即時送信
    yield f"data: {json.dumps({'feedback_id': feedback_id})}\n\n"

    try:
        # 2. システムヘルスチェック
        if not all([core_database.db_client, GEMINI_API_KEY]):
            logging.error("DBクライアントまたはAPIキーが設定されていません。")
            yield f"data: {json.dumps({'content': 'システムが利用できません。管理者にお問い合わせください。'})}\n\n"
            return

        # 3. ベクトル検索 (Stage 1: ドキュメント検索)
        search_results: List[Dict[str, Any]] = []
        relevant_docs: List[Dict[str, Any]] = []

        try:
            # 3a. 質問をベクトル化
            query_embedding_response = genai.embed_content(
                model=chat_req.embedding_model,
                content=user_input
            )
            query_embedding = query_embedding_response["embedding"] # 変数に格納

            # 3b. ベクトルDB検索
            if core_database.db_client:
                search_results = core_database.db_client.search_documents_by_vector(
                    collection_name=chat_req.collection,
                    embedding=query_embedding,
                    match_count=chat_req.top_k
                )
            
            # 3c. ログ出力 (フィルタリング前)
            # ▼▼▼ [ここから変更] .info -> .critical に変更 ▼▼▼
            # 3c. ログ出力 (フィルタリング前)
            # ▼▼▼ [ここから変更] .info に変更 ▼▼▼
            if search_results:
                logging.info(f"--- Stage 1 検索候補 (上位 {len(search_results)}件) ---")
                for doc in search_results:
                    doc_id = doc.get('id', 'N/A')
                    doc_source = doc.get('metadata', {}).get('source', 'N/A')
                    doc_similarity = doc.get('similarity', 0)
                    doc_content_preview = doc.get('content', '')[:50].replace('\n', ' ') + "..."
                    logging.info(f"  [ID: {doc_id}] [Sim: {doc_similarity:.4f}] (Source: {doc_source}) Content: '{doc_content_preview}'")
            else:
                logging.info(f"--- Stage 1 検索候補 (0件) ---")
            # ▲▲▲ [ここまで変更] ▲▲▲

        except Exception as e:
            logging.error(f"ベクトル検索エラー: {e}", exc_info=True)
            search_results = []
            logging.info(f"ベクトル化または検索に失敗したため、Stage 2へ移行します。") # <-- 変更# <-- 変更

        # 3d. 類似度によるフィルタリング
        strict_docs = [d for d in search_results if d.get('similarity', 0) >= STRICT_THRESHOLD]
        related_docs = [d for d in search_results if RELATED_THRESHOLD <= d.get('similarity', 0) < STRICT_THRESHOLD]
        relevant_docs = strict_docs + related_docs # 類似度が高いものを優先

        # 3e. ログ出力 (フィルタリング後)
        # ▼▼▼ [ここから変更] .info -> .critical に変更 ▼▼▼
        # 3e. ログ出力 (フィルタリング後)
        # ▼▼▼ [ここから変更] .info に変更 ▼▼▼
        if relevant_docs:
            logging.info(f"--- Stage 1 コンテキストに使用 (上記候補から {len(relevant_docs)}件を抽出) ---")
            
            for doc in relevant_docs:
                doc_id = doc.get('id', 'N/A')
                doc_source = doc.get('metadata', {}).get('source', 'N/A')
                doc_similarity = doc.get('similarity', 0)
                content_to_log = doc.get('metadata', {}).get('parent_content', doc.get('content', ''))
                doc_content_preview = content_to_log[:60].replace('\n', ' ') + "..."
                
                logging.info(f"  -> [使用] [ID: {doc_id}] [Sim: {doc_similarity:.4f}] (Source: {doc_source}) Content: '{doc_content_preview}'")

        else:
            logging.info(f"--- Stage 1 関連文書なし (閾値 {RELATED_THRESHOLD} 未満)。Stage 2へ移行します。 ---")
        # ▲▲▲ [ここまで変更] ▲▲▲
        # ▲▲▲ [ここまで変更] ▲▲▲


        # 4. 回答生成
        if relevant_docs:
            # --- Stage 1 RAG (ドキュメントに基づく回答) ---
            
            # 4a. コンテキストの構築
            context_parts = []
            current_char_length = 0
            
            for d in relevant_docs:
                source_name = d.get('metadata', {}).get('source', '不明')
                display_source = '履修要項2024' if source_name == 'output_gakubu.txt' else source_name
                parent_text = d.get('metadata', {}).get('parent_content', d.get('content', ''))

                if current_char_length + len(parent_text) > MAX_CONTEXT_CHAR_LENGTH and context_parts:
                    logging.warning(f"コンテキスト長が上限 ({MAX_CONTEXT_CHAR_LENGTH}文字) を超えるため、{len(relevant_docs) - len(context_parts)}件の文書をスキップします。")
                    break
                
                context_parts.append(f"<document source='{display_source}'>{parent_text}</document>")
                current_char_length += len(parent_text)
            
            context = "\n\n".join(context_parts)

            # 4b. プロンプトの構築 (修正済みのルール#7を含む)
            prompt = f"""あなたは札幌学院大学の学生サポートAIです。  
以下のルールに従ってユーザーの質問に答えてください。

# ルール
1. 回答は <context> 内の情報(大学公式情報)を**最優先**にしてください。
2. <context> に質問と「完全に一致する答え」が見つからない場合でも、「関連する可能性のある情報」が見つかった場合は、その情報を回答してください。
3. (ルール#2 に基づき)関連情報で回答した場合は、回答の最後に必ず以下の「注意書き」を加えてください。
   「※これは関連情報であり、ご質問NのG]意と完全に一致しない可能性があります。詳細は大学の公式窓口にご確認ください。」
4. 出典を引用する場合は、使用した情報の直後に `[出典: ...]` を付けてください。
5. **大学固有の事実（学費、特定のゼミ、手続き、校舎の場所など）を推測して答えてはいけません。**
6. **特に重要**: <context> 内の情報を使って回答することを最優先にしてください。ただし、<context> 内のどの情報も質問と全く関連性がないと判断した場合に限り、「ご質問いただいた内容については、関連する情報が見つかりませんでした。お手数ですが、大学の公式サイトをご確認いただくか、窓口までお問い合わせください。」と回答しても構いません。
7. **一般知識の使用について**:
   - あなたの知識は、<context> の情報を**簡潔にまとめる（要約する）**ため**だけ**に使用してください。
   - <context> の情報を補足するために、<context> に書かれていない情報を付け加えてはいけません。

# 出力形式
- 学生に分かりやすい「です・ます調」で回答すること。
- 箇条書きや見出しを活用して整理すること。
- <context> 内にURLがあれば「参考URL:」として末尾にまとめること。その際、必ず **Markdown 形式（例: `[リンクテキスト](URL)`）** を使用すること。

<context>
{context}
</context>

<query>
{user_input}
</query>

---
[あなたの回答]
回答:
"""
            # 4c. 安全フィルターの無効化設定 (修正済み)
            # ↓↓↓ [修正] HARM_CATEGORY_CIVIC_INTEGRITY の行を削除
            safety_settings = {
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            # ↑↑↑ [修正]
            
            logging.info(f"--- APIに送信するプロンプト (Stage 1 RAG) ---\n(Context文字数: {len(context)}) \n--- プロンプト終了 ---")
            
            model = genai.GenerativeModel(
                chat_req.model,
                safety_settings=safety_settings
            )
            
            # 4d. ストリーミング回答の生成
            response_text = ""
            full_response = ""
            try:
                stream = await safe_generate_content(model, prompt, stream=True)
                async for chunk in stream:
                    # ▼▼▼ [ここから修正] ▼▼▼
                    try:
                        if chunk.text:
                            response_text += chunk.text
                            # ストリーミングでクライアントに送信
                            yield f"data: {json.dumps({'content': chunk.text})}\n\n"
                    except ValueError:
                        # .text が存在しないチャンク (finish_reason=STOP の最後の空チャンクなど) は無視
                        pass
                    # ▲▲▲ [ここまで修正] ▲▲▲
                
                full_response = format_urls_as_links(response_text.strip() or "回答を生成できませんでした。")

            except Exception as e:
                logging.error(f"Stage 1 回答生成エラー: {e}", exc_info=True)
                full_response = "回答の生成中にエラーが発生しました。" 
                
                if "StopAsyncIteration" in str(e) or "空の応答" in str(e):
                    logging.warning("セーフティフィルター作動の可能性があるため、Stage 2に移行します。")
                    full_response = AI_NOT_FOUND_MESSAGE 
                else:
                    # 400 InvalidArgument など、その他のエラー
                    yield f"data: {json.dumps({'content': full_response})}\n\n"
            
            
            # -----------------------------------------------
            # AIの回答をチェックして、Stage 2 に移行するか判断
            # -----------------------------------------------
            
            if "エラーが発生しました" not in full_response:
                if AI_NOT_FOUND_MESSAGE in full_response:
                    logging.info("Stage 1 AIがコンテキスト内に回答を発見できませんでした。Stage 2 (Q&A) に移行します。")
                    # Stage 2 を実行
                    async for fallback_chunk in _run_stage2_fallback(query_embedding, session_id, user_input, feedback_id):
                        yield fallback_chunk
                
                else:
                    # --- Stage 1 成功 ---
                    # 履歴に追加
                    history_manager.add_to_history(session_id, "user", user_input)
                    history_manager.add_to_history(session_id, "assistant", full_response)

                    # 5. 最終処理 (フィードバック表示トリガー)
                    yield f"data: {json.dumps({'show_feedback': True, 'feedback_id': feedback_id})}\n\n"
            else:
                pass


        else:
            # --- Stage 1 RAG (最初から文書が見つからなかった場合) ---
            # Stage 2 関数を呼び出す
            async for fallback_chunk in _run_stage2_fallback(query_embedding, session_id, user_input, feedback_id):
                yield fallback_chunk

    except Exception as e:
        err_msg = f"エラーが発生しました: {e}"
        logging.error(f"チャットロジック全体のエラー: {e}", exc_info=True)
        yield f"data: {json.dumps({'content': err_msg})}\n\n"