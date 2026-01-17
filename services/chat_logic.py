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

# エラーメッセージ
AI_MESSAGES = {
    "NOT_FOUND": (
        "申し訳ありません。ご質問に関して、学内の公式ドキュメントやデータベースから"
        "確実な根拠を見つけることができませんでした。"
        "不正確な回答を避けるため、大学窓口へ直接お問い合わせいただくことをお勧めします。"
    ),
    "RATE_LIMIT": "現在アクセスが集中しており、回答生成に時間がかかっています。1分ほど待ってから再度お試しください。",
    "SYSTEM_ERROR": "システムエラーが発生しました。しばらく時間をおいて再度お試しください。",
    "BLOCKED": "生成された回答がセーフティガイドラインに抵触したため、表示できませんでした。"
}

# スレッドプール
executor = ThreadPoolExecutor(max_workers=4)

# -----------------------------------------------------------------------------
# 2. プロンプト & スキーマ定義
# -----------------------------------------------------------------------------

# リランク出力用の型定義
class RankedItem(typing.TypedDict):
    id: int
    score: float
    reason: str

class RerankResponse(typing.TypedDict):
    ranked_items: list[RankedItem]

# リランク用プロンプト
PROMPT_RERANK = """
あなたは大学の学生課スタッフです。ユーザーの質問に対し、以下のドキュメントが回答根拠として適切か0-10点で厳密に採点してください。

評価基準:
- 10点: 質問の核心的な答え（日付、金額、手順など）が直接書かれている。
- 5-9点: 部分的に関連する情報、または答えを導き出すための前提知識が含まれる。
- 0-4点: キーワードは一致するが、文脈が異なる。

質問: {query}

候補ドキュメント:
{candidates_text}
"""

# マルチクエリ生成用プロンプト
PROMPT_MULTI_QUERY = """
ユーザーの質問に対して、データベース検索の網羅性を高めるための「3つの異なる検索クエリ」を作成してください。
以下の観点でクエリを作成し、Pythonのリスト形式 ["query1", "query2", "query3"] のみを出力してください。

1. **キーワード検索用**: 質問に含まれる重要単語（名詞）の羅列
2. **意味検索用（具体的）**: 質問の意図を汲み取り、より具体的にした文章
3. **関連語検索用**: 専門用語や類義語、言い換え表現を含めた文章

質問: "{user_query}"
"""

# 回答生成用システムプロンプト
PROMPT_SYSTEM_GENERATION = """
あなたは札幌学院大学の学生サポートAIです。
提供された <context> 内の情報**のみ**を使用して、質問に回答してください。

# 重要なルール（厳守）
1. **思考プロセス（Chain of Thought）**:
   回答を出力する前に、必ず「質問の分析」「関連情報の抽出」「矛盾の確認」を内部的に行ってください。
   文脈が不明瞭な場合は、推測せず「情報が見つかりません」と判断してください。

2. **根拠の明示**:
   回答する全ての事実について、根拠となるドキュメントIDを文末に `[1]` の形式で付記してください。
   例: 「授業料の納入期限は5月末です[1]。」
   
3. **トーン & マナー**:
   - 学生に寄り添った、丁寧で親しみやすい「です・ます」調。
   - 結論を先に述べる（PREP法）。
   - 重要な日付、金額、場所は**太字**にする。
"""

# -----------------------------------------------------------------------------
# 3. ユーティリティ関数
# -----------------------------------------------------------------------------
def get_or_create_session_id(request: Request) -> str:
    session_id = request.session.get('chat_session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session['chat_session_id'] = session_id
    return session_id

def log_context(session_id: str, message: str, level: str = "info"):
    msg = f"[Session: {session_id}] {message}"
    getattr(logging, level, logging.info)(msg)

def send_sse(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

async def api_request_with_retry(func, *args, **kwargs):
    max_retries = 3
    default_delay = 2
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "Quota" in error_str:
                if attempt == max_retries - 1:
                    logging.error(f"API Quota Exceeded after {max_retries} retries.")
                    raise e
                wait_time = default_delay * (2 ** attempt)
                logging.warning(f"Rate limit hit. Waiting {wait_time:.1f}s. Retrying...")
                await asyncio.sleep(wait_time)
            else:
                raise e

class ChatHistoryManager:
    def __init__(self):
        self._histories: Dict[str, List[Dict[str, str]]] = {}

    def add(self, session_id: str, role: str, content: str):
        if session_id not in self._histories:
            self._histories[session_id] = []
        self._histories[session_id].append({"role": role, "content": content})
        if len(self._histories[session_id]) > PARAMS["MAX_HISTORY_LENGTH"]:
            self._histories[session_id] = self._histories[session_id][-PARAMS["MAX_HISTORY_LENGTH"]:]

history_manager = ChatHistoryManager()

# -----------------------------------------------------------------------------
# 4. コアロジック: 高度な検索パイプライン
# -----------------------------------------------------------------------------
class SearchPipeline:
    @staticmethod
    async def generate_multi_queries(user_query: str) -> List[str]:
        prompt = PROMPT_MULTI_QUERY.format(user_query=user_query)
        try:
            model = genai.GenerativeModel(USE_MODEL)
            resp = await api_request_with_retry(
                model.generate_content_async, prompt, safety_settings=SAFETY_SETTINGS
            )
            match = re.search(r'\[.*\]', resp.text, re.DOTALL)
            if match:
                queries = json.loads(match.group())
                return [q for q in queries if isinstance(q, str)]
            return [user_query]
        except Exception as e:
            logging.warning(f"Multi-query generation failed: {e}")
            return [user_query]

    @staticmethod
    async def rerank(query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        if not documents:
            return []
        
        candidates_text = ""
        for i, doc in enumerate(documents):
            meta = doc.get('metadata', {})
            content = doc.get('content', '')
            candidates_text += f"ID:{i} [Source:{meta.get('source', '?')}]\n{content}\n\n"

        formatted_prompt = PROMPT_RERANK.format(query=query, candidates_text=candidates_text)

        try:
            model = genai.GenerativeModel(USE_MODEL)
            resp = await api_request_with_retry(
                model.generate_content_async,
                formatted_prompt,
                generation_config=GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=RerankResponse
                ),
                safety_settings=SAFETY_SETTINGS
            )
            
            data = json.loads(resp.text)
            reranked = []
            seen_indices = set()

            for item in data.get("ranked_items", []):
                idx = item.get("id")
                score = item.get("score")
                if idx is not None and 0 <= idx < len(documents) and idx not in seen_indices:
                    if score >= PARAMS["RERANK_SCORE_THRESHOLD"]:
                        doc = documents[idx]
                        doc['rerank_score'] = score
                        doc['rerank_reason'] = item.get("reason", "")
                        reranked.append(doc)
                        seen_indices.add(idx)
            
            reranked.sort(key=lambda x: x['rerank_score'], reverse=True)
            return reranked[:top_k]
        except Exception as e:
            logging.error(f"Rerank Error: {e}")
            return documents[:top_k]

    @staticmethod
    async def filter_diversity(documents: List[Dict], threshold: float = 0.65) -> List[Dict]:
        loop = asyncio.get_running_loop()
        unique_docs = []
        def _calc_sim(a, b):
            return SequenceMatcher(None, a, b).ratio()
        for doc in documents:
            content = doc.get('content', '')
            is_duplicate = False
            for selected in unique_docs:
                sim = await loop.run_in_executor(executor, _calc_sim, content, selected.get('content', ''))
                if sim > threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_docs.append(doc)
        return unique_docs

    @staticmethod
    def reorder_documents(documents: List[Dict]) -> List[Dict]:
        if not documents:
            return []
        first_half = documents[0::2]
        second_half = documents[1::2][::-1]
        return first_half + second_half

def _build_references(response_text: str, sources_map: Dict[int, str]) -> str:
    unique_refs = []
    seen_sources = set()
    cited_ids = set(map(int, re.findall(r'\[(\d+)\]', response_text)))
    for idx, src in sources_map.items():
        if src in seen_sources: continue
        if idx in cited_ids or idx <= 2:
            unique_refs.append(f"* [{idx}] {src}")
            seen_sources.add(src)
    if unique_refs:
        return "\n\n### 参照元データ\n" + "\n".join(unique_refs)
    return ""

# -----------------------------------------------------------------------------
# 5. メイン: チャットロジック
# -----------------------------------------------------------------------------
async def enhanced_chat_logic(request: Request, chat_req: ChatQuery):
    session_id = get_or_create_session_id(request)
    feedback_id = str(uuid.uuid4())
    user_input = chat_req.query.strip()
    full_resp = ""
    
    # ---------------------------------------------------------
    # 0.0s: 初期メッセージ
    # ---------------------------------------------------------
    yield send_sse({'feedback_id': feedback_id, 'status_message': '🤔 質問の意図を分解しています...'})

    try:
        # ---------------------------------------------------------
        # Step 1: Multi-Query Generation
        # ---------------------------------------------------------
        queries = await SearchPipeline.generate_multi_queries(user_input)
        queries.append(user_input)
        queries = list(set(queries))
        
        # ---------------------------------------------------------
        # 1.5s: 検索開始メッセージ
        # ---------------------------------------------------------
        # 実際のクエリ数に合わせて表示（例: "3つの視点..."）
        yield send_sse({'status_message': f'📚 {len(queries)}つの視点でデータベースを横断検索中...'})

        # ---------------------------------------------------------
        # Step 2: Parallel Hybrid Search
        # ---------------------------------------------------------
        all_raw_docs = []
        embedding_tasks = [
            genai.embed_content_async(
                model=chat_req.embedding_model,
                content=q,
                task_type="retrieval_query"
            ) for q in queries
        ]
        embeddings_results = await asyncio.gather(*embedding_tasks)
        
        for q, raw_emb_result in zip(queries, embeddings_results):
            query_embedding = raw_emb_result["embedding"]
            
            # QA DBチェック（オリジナルのみ）
            if q == user_input:
                if qa_hits := core_database.db_client.search_fallback_qa(query_embedding, match_count=1):
                    top_qa = qa_hits[0]
                    if top_qa.get('similarity', 0) >= PARAMS["QA_SIMILARITY_THRESHOLD"]:
                        resp = format_urls_as_links(f"よくあるご質問に回答が見つかりました。\n\n---\n{top_qa['content']}")
                        history_manager.add(session_id, "assistant", resp)
                        yield send_sse({'content': resp, 'show_feedback': True, 'feedback_id': feedback_id})
                        return

            # ドキュメント検索
            docs = core_database.db_client.search_documents_hybrid(
                collection_name=chat_req.collection,
                query_text=q,
                query_embedding=query_embedding,
                match_count=15 
            )
            all_raw_docs.extend(docs)

        if not all_raw_docs:
            yield send_sse({'content': AI_MESSAGES["NOT_FOUND"]})
            return

        # ---------------------------------------------------------
        # Step 3: Filtering & Reranking
        # ---------------------------------------------------------
        unique_docs = await SearchPipeline.filter_diversity(all_raw_docs)
        
        # ---------------------------------------------------------
        # 2.5s: 精読メッセージ
        # ---------------------------------------------------------
        # ヒットした実際の件数を表示
        yield send_sse({'status_message': f'🧐 ヒットした{len(unique_docs)}件の文献を精読しています...'})
        
        reranked_docs = await SearchPipeline.rerank(user_input, unique_docs[:25], top_k=8)
        relevant_docs = SearchPipeline.reorder_documents(reranked_docs)

        if not relevant_docs:
            yield send_sse({'content': AI_MESSAGES["NOT_FOUND"]})
            return

        # ---------------------------------------------------------
        # 6.0s: 検証・回答作成メッセージ
        # ---------------------------------------------------------
        yield send_sse({'status_message': '✍️ 情報に矛盾がないか検証し、回答を作成中...'})
        
        # ---------------------------------------------------------
        # Step 4: Generation
        # ---------------------------------------------------------
        context_parts = []
        sources_map = {}
        for idx, doc in enumerate(relevant_docs, 1):
            src = doc.get('metadata', {}).get('source', '不明')
            title = doc.get('metadata', {}).get('title', '')
            sources_map[idx] = f"{title} ({src})"
            context_parts.append(f"<doc id='{idx}' source='{src}'>\n{doc.get('content','')}\n</doc>")
        
        context_str = "\n".join(context_parts)
        full_system_prompt = f"{PROMPT_SYSTEM_GENERATION}\n<context>\n{context_str}\n</context>"

        model = genai.GenerativeModel(USE_MODEL)
        stream = await api_request_with_retry(
            model.generate_content_async,
            [full_system_prompt, f"質問: {user_input}"],
            stream=True,
            safety_settings=SAFETY_SETTINGS
        )
        
        accumulated_text = ""
        # ---------------------------------------------------------
        # 8.0s: 回答ストリーミング開始
        # ---------------------------------------------------------
        async for chunk in stream:
            # ★修正: 空のチャンク（テキストを含まない完了信号など）によるエラーを回避
            try:
                # chunk.text にアクセスするだけで検証が行われるため、try-exceptで囲む
                if chunk.text:
                    accumulated_text += chunk.text
                    yield send_sse({'content': chunk.text})
            except Exception:
                # テキストが含まれていないチャンク（メタデータのみ等）は無視して次へ
                pass
        
        full_resp = accumulated_text
        
        if not full_resp:
             yield send_sse({'content': AI_MESSAGES["BLOCKED"]})
             return

        if "情報が見つかりません" not in full_resp:
            refs_text = _build_references(full_resp, sources_map)
            if refs_text:
                yield send_sse({'content': refs_text})
                full_resp += refs_text
        
        history_manager.add(session_id, "assistant", full_resp)

    except Exception as e:
        log_context(session_id, f"Pipeline Error: {e}", "error")
        if not full_resp:
            yield send_sse({'content': AI_MESSAGES["SYSTEM_ERROR"]})
            
    finally:
        yield send_sse({'show_feedback': True, 'feedback_id': feedback_id})

# -----------------------------------------------------------------------------
# 6. 分析機能
# -----------------------------------------------------------------------------
async def analyze_feedback_trends(logs: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
    if not logs:
        yield send_sse({'content': '分析対象データがありません。'})
        return
    summary = "\n".join([f"- 評価:{l.get('rating','-')} | {l.get('comment','-')[:100]}" for l in logs[:50]])
    prompt = f"以下のログを分析してください:\n{summary}"
    try:
        model = genai.GenerativeModel(USE_MODEL)
        stream = await api_request_with_retry(model.generate_content_async, prompt, stream=True)
        async for chunk in stream:
            if chunk.text:
                yield send_sse({'content': chunk.text})
    except Exception as e:
        yield send_sse({'content': f'分析エラー: {e}'})