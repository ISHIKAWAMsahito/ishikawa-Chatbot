import json
import logging
import asyncio
from typing import List, Dict, Optional, Any
from difflib import SequenceMatcher
from collections import deque

import google.generativeai as genai
import typing_extensions as typing
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, AsyncRetrying, RetryError
from async_lru import alru_cache
from langsmith import traceable

from services.llm import LLMService
from services import prompts
from core.constants import PARAMS
from core.database import db_client
from core.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

# Gemini API設定
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class RankedItem(typing.TypedDict):
    id: int
    score: float
    reason: str

class RerankResponse(typing.TypedDict):
    ranked_items: list[RankedItem]

class SearchService:
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    async def get_embedding(self, text: str, model: str = "models/gemini-embedding-001") -> List[float]:
        """テキストをベクトル化する"""
        try:
            result = genai.embed_content(
                model=model,
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise e

    # ----------------------------------------------------------------
    # Step 1: クエリ拡張 (高速化: 思考プロセス省略)
    # ----------------------------------------------------------------
    @alru_cache(maxsize=100)
    @traceable(name="Step1_Query_Expansion", run_type="chain")
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=10), # 待機時間を短縮 [cite: 21]
        stop=stop_after_attempt(3), 
        reraise=True 
    )
    async def expand_query(self, query: str) -> str:
        """
        ユーザーの質問を、検索ヒット率が高まるキーワード・フレーズに変換する。
        """
        cot_prompt = prompts.QUERY_EXPANSION.format(query=query)
        try:
            response = await self.llm.generate_stream(cot_prompt)
            full_text = ""
            async for chunk in response:
                if chunk.text: full_text += chunk.text
            return full_text.strip()
        except Exception as e:
            logger.warning(f"Query expansion error: {e}")
            return query # 失敗時は元のクエリを使用

    # ----------------------------------------------------------------
    # Step 2: AIリランク (精度向上: 上位5件精査・6点切り捨て)
    # ----------------------------------------------------------------
    @traceable(name="Step2_AI_Rerank", run_type="chain")
    async def rerank(self, query: str, documents: List[Dict], top_k: int) -> List[Dict]:
        """AIを用いて検索結果を再ランク付けする"""
        if not documents:
            return []

        # 処理量削減のため上位5件に絞る [cite: 25]
        initial_candidates = documents[:5] 

        candidates_text = ""
        for i, doc in enumerate(initial_candidates):
            meta = doc.get('metadata', {})
            # 親コンテンツ（全文）を評価に使用 [cite: 26]
            content = meta.get('parent_content', doc.get('content', ''))
            snippet = content[:300].replace('\n', ' ')
            candidates_text += f"Document ID: {i}\nSource: {meta.get('source', 'Unknown')}\nContent: {snippet}...\n\n"

        rerank_prompt = prompts.RERANK.format(query=query, candidates_text=candidates_text)

        try:
            # 待機時間を短縮してリトライ [cite: 21]
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(Exception),
                wait=wait_exponential(multiplier=1, min=1, max=5), 
                stop=stop_after_attempt(3),
                reraise=True
            ):
                with attempt:
                    resp = await self.llm.generate_json(rerank_prompt, RerankResponse)
                    raw_json = resp.text if hasattr(resp, 'text') else str(resp)
                    raw_json = raw_json.replace("```json", "").replace("```", "").strip()
                    data = json.loads(raw_json)
                    
                    reranked_docs = []
                    for item in data.get("ranked_items", []):
                        idx = item.get("id")
                        score = float(item.get("score", 0.0))
                        reason = item.get("reason", "")
                        
                        if idx is not None and isinstance(idx, int) and 0 <= idx < len(initial_candidates):
                            # 6点未満は切り捨て [cite: 60]
                            if score >= 6.0: 
                                doc = initial_candidates[idx].copy()
                                # 全文情報の保持
                                if 'parent_content' in doc.get('metadata', {}):
                                    doc['content'] = doc['metadata']['parent_content']
                                doc['rerank_score'] = score
                                doc['rerank_reason'] = reason
                                reranked_docs.append(doc)

                    reranked_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
                    return reranked_docs[:top_k]

        except Exception as e:
            logger.error(f"Rerank failed: {e}. Fallback to vector order.")
            return self._fallback_docs(documents, top_k)

    def _fallback_docs(self, documents: List[Dict], top_k: int) -> List[Dict]:
        """エラー時のフォールバック処理"""
        fallback_docs = []
        for doc in documents[:top_k]:
            d = doc.copy()
            if 'parent_content' in d.get('metadata', {}):
                d['content'] = d['metadata']['parent_content']
            fallback_docs.append(d)
        return fallback_docs

    # ----------------------------------------------------------------
    # Step 3: LitM対策 (U字型配置)
    # ----------------------------------------------------------------
    def reorder_litm(self, documents: List[Dict]) -> List[Dict]:
        """重要情報を先頭と末尾に配置 """
        if not documents: return []
        dq = deque(documents)
        reordered = []
        if dq: reordered.append(dq.popleft()) # 1位
        temp_tail = []
        while dq:
            temp_tail.append(dq.popleft())
            if dq:
                reordered.append(dq.popleft())
        return reordered + temp_tail[::-1]

    # ----------------------------------------------------------------
    # Step 4: 多様性フィルタリング
    # ----------------------------------------------------------------
    def filter_diversity(self, documents: List[Dict], threshold: float = 0.7) -> List[Dict]:
        """重複度70%以上をカット """
        unique_docs = []
        for doc in documents:
            content = doc.get('content', '')
            is_duplicate = False
            for selected in unique_docs:
                sim = SequenceMatcher(None, content, selected.get('content', '')).ratio()
                if sim > threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_docs.append(doc)
        return unique_docs

    # ----------------------------------------------------------------
    # 統合検索メソッド (パイプライン実行)
    # ----------------------------------------------------------------
    @traceable(name="Search_Pipeline", run_type="chain")
    async def search(self, query: str, session_id: str, collection_name: str, top_k: int = 5, embedding_model: str = None) -> Dict[str, Any]:
        try:
            # 1. Embedding生成
            model = embedding_model or "models/gemini-embedding-001"
            query_embedding = await self.get_embedding(query, model=model)

            if not db_client.client:
                return {"documents": []}

            # ハイブリッド検索の準備 
            # ベクトル検索とキーワード検索を併用
            match_count = top_k * 3 # リランク用に多めに取得
            if match_count < 10: match_count = 10

            params = {
                "p_query_text": query,          # キーワード
                "p_query_embedding": query_embedding, # ベクトル
                "p_match_count": match_count,
                "p_collection_name": collection_name
            }
            
            # ハイブリッド関数呼び出し (なければベクトル検索へフォールバック)
            try:
                response = db_client.client.rpc("match_documents_hybrid", params).execute()
            except Exception:
                vector_params = {
                    "p_query_embedding": query_embedding,
                    "p_match_count": match_count,
                    "p_collection_name": collection_name
                }
                response = db_client.client.rpc("match_documents", vector_params).execute()

            documents = response.data if response.data else []
            if not documents:
                return {"documents": []}

            # 3. Rerank (上位5件精査)
            reranked_docs = await self.rerank(query, documents, top_k)

            # 4. LitM Reorder
            reordered_docs = self.reorder_litm(reranked_docs)
            
            # 5. Diversity Filter
            final_docs = self.filter_diversity(reordered_docs, threshold=0.7)

            return {"documents": final_docs}

        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            return {"documents": []}