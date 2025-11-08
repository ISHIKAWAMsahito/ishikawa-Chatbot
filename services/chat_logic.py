import logging
import uuid
import json
import asyncio
from collections import defaultdict
from typing import List, Dict
from fastapi import Request, HTTPException
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold

from core.config import GEMINI_API_KEY

# ↓↓↓ [修正] 変数ではなくモジュールをインポートし、別名を付ける
from core import database as core_database
# ↑↑↑ [修正]

from models.schemas import ChatQuery
from services.utils import format_urls_as_links

# チャット履歴の管理
chat_histories: Dict[str, List[Dict[str, str]]] = defaultdict(list)
MAX_HISTORY_LENGTH = 20

def get_or_create_session_id(request: Request) -> str:
    """セッションIDを取得または新規作成"""
    session_id = request.session.get('chat_session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session['chat_session_id'] = session_id
    return session_id

def add_to_history(session_id: str, role: str, content: str):
    """チャット履歴に追加"""
    # 一時的に無効化（会話の記録を完全に停止）
    return

def get_history(session_id: str) -> List[Dict[str, str]]:
    """チャット履歴を取得"""
    return chat_histories.get(session_id, [])

def clear_history(session_id: str):
    """チャット履歴をクリア"""
    if session_id in chat_histories:
        del chat_histories[session_id]

async def safe_generate_content(model, prompt, stream=False, max_retries=3):
    """レート制限を考慮した安全なコンテンツ生成"""
    import re
    
    for attempt in range(max_retries):
        try:
            if stream:
                return await model.generate_content_async(
                    prompt,
                    stream=True,
                    generation_config=GenerationConfig(
                        max_output_tokens=1024,
                        temperature=0.1
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

async def enhanced_chat_logic(request: Request, chat_req: ChatQuery):
    """RAG + フォールバック対応のチャット処理"""
    user_input = chat_req.query.strip()
    feedback_id = str(uuid.uuid4())
    session_id = get_or_create_session_id(request)

    yield f"data: {json.dumps({'feedback_id': feedback_id})}\n\n"

    try:
        # ↓↓↓ [修正] core_database を参照
        if not all([core_database.db_client, GEMINI_API_KEY]):
            yield f"data: {json.dumps({'content': 'システムが利用できません。管理者にお問い合わせください。'})}\n\n"
            return
        # ↑↑↑ [修正]

        # ベクトル検索処理
        STRICT_THRESHOLD = 0.80
        RELATED_THRESHOLD = 0.75
        search_results = []
        relevant_docs = []

        try:
            query_embedding_response = genai.embed_content(
                model=chat_req.embedding_model,
                content=user_input
            )
            query_embedding = query_embedding_response["embedding"]

            # ↓↓↓ [修正] core_database を参照
            if core_database.db_client:
                search_results = core_database.db_client.search_documents_by_vector(
                    collection_name=chat_req.collection,
                    embedding=query_embedding,
                    match_count=chat_req.top_k
                )
            # ↑↑↑ [修正]

            logging.info(f"検索結果件数: {len(search_results)}")

        except Exception as e:
            logging.error(f"ベクトル検索エラー: {e}")
            search_results = []

        # 類似度フィルタリング
        strict_docs = [d for d in search_results if d.get('similarity', 0) >= STRICT_THRESHOLD]
        related_docs = [d for d in search_results if RELATED_THRESHOLD <= d.get('similarity', 0) < STRICT_THRESHOLD]
        relevant_docs = strict_docs + related_docs

        # ログ出力
        if relevant_docs:
            logging.info(f"--- Stage 1 RAG ヒット (上位 {len(relevant_docs)}件) ---")
            for doc in relevant_docs:
                doc_id = doc.get('id', 'N/A')
                doc_source = doc.get('metadata', {}).get('source', 'N/A')
                doc_similarity = doc.get('similarity', 0)
                doc_content_preview = doc.get('content', '')[:50].replace('\n', ' ') + "..."
                
                logging.info(f"  [ID: {doc_id}] [Sim: {doc_similarity:.4f}] (Source: {doc_source}) Content: '{doc_content_preview}'")

        # コンテキスト生成と回答生成
        if relevant_docs:
            context_parts = []
            
            for d in relevant_docs:
                source_name = d.get('metadata', {}).get('source', '不明')
                
                # ファイル名のマッピング
                if source_name == 'output_gakubu.txt':
                    display_source = '履修要項2024'
                else:
                    display_source = source_name
                
                context_parts.append(
                    f"<document source='{display_source}'>{d.get('content', '')}</document>"
                )
            
            context = "\n\n".join(context_parts)

            prompt = f"""あなたは札幌学院大学の学生サポートAIです。  
以下のルールに従ってユーザーの質問に答えてください。

# ルール
1. 回答は <context> 内の情報(大学公式情報)のみに基づいてください。
2. <context> に質問と「完全に一致する答え」が見つからない場合でも、「関連する可能性のある情報」(例:質問は「大会での欠席」だが、資料には「病欠」について記載がある場合)が見つかった場合は、その情報を回答してください。
3. (ルール#2 に基づき)関連情報で回答した場合は、回答の最後に必ず以下の「注意書き」を加えてください。
   「※これは関連情報であり、ご質問の意図と完全に一致しない可能性があります。詳細は大学の公式窓口にご確認ください。」
4. 出典を引用する場合は、使用した情報の直後に `[出典: ...]` を付けてください。
5. 大学固有の情報を推測して答えてはいけません。
6. **特に重要**: <context> には必ず関連情報が含まれています。その情報を使って回答すること。「見つかりませんでした」と答えてはいけません。

# 出力形式
- 学生に分かりやすい「です・ます調」で回答すること。
- 箇条書きや見出しを活用して整理すること。
- <context> 内にURLがあれば「参考URL:」として末尾にまとめること。

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

            # 安全フィルター無効化設定
            safety_settings = {
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            
            model = genai.GenerativeModel(
                chat_req.model,
                safety_settings=safety_settings
            )
            response_text = ""
            try:
                stream = await safe_generate_content(model, prompt, stream=True)
                async for chunk in stream:
                    if chunk.text:
                        response_text += chunk.text
            except Exception as e:
                logging.error(f"生成エラー: {e}")
                response_text = "回答の生成中にエラーが発生しました。"

            full_response = format_urls_as_links(response_text.strip() or "回答を生成できませんでした。")
            
            add_to_history(session_id, "user", user_input)
            add_to_history(session_id, "assistant", response_text)
            yield f"data: {json.dumps({'content': full_response})}\n\n"

        else:
            # フォールバック処理 (Stage 2: Q&Aベクトル検索)
            logging.info(f"Stage 1 RAG 失敗。Stage 2 (Q&Aベクトル検索) を実行します。")

            try:
                # ↓↓↓ [修正] core_database を参照
                fallback_results = core_database.db_client.search_fallback_qa(
                    embedding=query_embedding,
                    match_count=1
                )
                # ↑↑↑ [修正]

                if fallback_results:
                    best_match = fallback_results[0]
                    FALLBACK_SIMILARITY_THRESHOLD = 0.59

                    if best_match.get('similarity', 0) >= FALLBACK_SIMILARITY_THRESHOLD:
                        logging.info(
                            f"Stage 2 RAG 成功。類似Q&Aを回答します (Similarity: {best_match['similarity']:.2f})"
                        )
                        fallback_response = f"""データベースに直接の情報は見つかりませんでしたが、関連する「よくあるご質問」がありましたのでご案内します。

---
{best_match['content']}
"""
                        full_response = format_urls_as_links(fallback_response)
                    else:
                        logging.info(
                            f"Stage 2 RAG 失敗。類似するQ&Aが見つかりませんでした (Best Similarity: {best_match.get('similarity', 0):.2f})"
                        )
                        fallback_response = "申し訳ありませんが、ご質問に関連する情報がデータベース(Q&Aを含む)に見つかりませんでした。大学公式サイトをご確認いただくか、学生支援課までお問い合わせください。"
                        full_response = format_urls_as_links(fallback_response)
                else:
                    logging.info("Stage 2 RAG 失敗。Q&Aデータベースが空か、検索エラーです。")
                    fallback_response = "申し訳ありませんが、ご質問に関連する情報が見つかりませんでした。大学公式サイトをご確認いただくか、学生支援課までお問い合わせください。"
                    full_response = format_urls_as_links(fallback_response)

            except Exception as e_fallback:
                logging.error(f"Stage 2 (Q&A検索) でエラーが発生: {e_fallback}")
                fallback_response = "申し訳ありません。現在、関連情報の検索中にエラーが発生しました。時間をおいて再度お試しください。"
                full_response = format_urls_as_links(fallback_response)

            add_to_history(session_id, "user", user_input)
            add_to_history(session_id, "assistant", fallback_response)

            yield f"data: {json.dumps({'content': full_response})}\n\n"

        # 最後のフィードバック出力
        yield f"data: {json.dumps({'show_feedback': True, 'feedback_id': feedback_id})}\n\n"

    except Exception as e:
        err = f"エラーが発生しました: {e}"
        logging.error(f"チャットロジック全体エラー: {err}")
        yield f"data: {json.dumps({'content': err})}\n\n"