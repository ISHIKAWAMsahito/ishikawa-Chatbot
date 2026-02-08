import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
from collections import deque

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, AsyncRetrying
from async_lru import alru_cache
from langsmith import traceable

# 必要なモジュールをインポート
from services.llm import LLMService
from services import prompts
from core.database import db_client
from core.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

# Gemini API設定
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class SearchService:
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    async def get_embedding(self, text: str, model: str = "models/gemini-embedding-001") -> List[float]:
        """テキストをベクトル化する"""
        try:
            return await self.llm.get_embedding(text, model)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []

    # ----------------------------------------------------------------
    # Step 1: クエリ拡張
    # ----------------------------------------------------------------
    @alru_cache(maxsize=100)
    @traceable(name="Step1_Query_Expansion", run_type="chain")
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        stop=stop_after_attempt(2), 
        reraise=False
    )
    async def expand_query(self, query: str) -> str:
        """ユーザーの質問を、検索ヒット率が高まるキーワードに変換"""
        cot_prompt = prompts.QUERY_EXPANSION.format(query=query)
        try:
            response = await self.llm.generate_stream(cot_prompt)
            full_text = ""
            async for chunk in response:
                if chunk.text: full_text += chunk.text
            
            expanded = full_text.strip()
            if not expanded:
                return query
            return expanded
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return query

    # ----------------------------------------------------------------
    # Step 2: AIリランク (並列処理版)
    # ----------------------------------------------------------------
    async def _check_relevance_single(self, query: str, doc: Dict, index: int) -> Optional[Dict]:
        """単一ドキュメントの関連性判定（並列実行用ヘルパー）"""
        meta = doc.get('metadata', {})
        content = meta.get('parent_content', doc.get('content', ''))
        snippet = content[:500].replace('\n', ' ')
        
        prompt = f"""
        あなたは検索エンジンのリランクシステムです。
        以下の「検索クエリ」に対して、「対象ドキュメント」がどれくらい関連しているか、0.0〜10.0のスコアで評価してください。
        
        検索クエリ: {query}
        
        対象ドキュメント:
        {snippet}
        
        出力は以下のJSON形式のみを返してください。
        {{
            "score": 8.5,
            "reason": "具体的なキーワードが含まれているため"
        }}
        """

        try:
            async for attempt in AsyncRetrying(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=3)):
                with attempt:
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    response = await model.generate_content_async(
                        prompt,
                        generation_config={"response_mime_type": "application/json"}
                    )
                    
                    text = response.text.replace("```json", "").replace("```", "").strip()
                    data = json.loads(text)
                    score = float(data.get("score", 0.0))
                    
                    if score >= 6.0:
                        doc_copy = doc.copy()
                        if 'parent_content' in meta:
                            doc_copy['content'] = meta['parent_content']
                        doc_copy['rerank_score'] = score
                        doc_copy['rerank_reason'] = data.get("reason", "")
                        return doc_copy
                    else:
                        return None
                        
        except Exception as e:
            logger.warning(f"Relevance check failed for doc {index}: {e}")
            return None

    @traceable(name="Step2_AI_Rerank", run_type="chain")
    async def rerank(self, query: str, documents: List[Dict], top_k: int) -> List[Dict]:
        """AIを用いて検索結果を再ランク付けする (並列実行)"""
        if not documents:
            return []

        initial_candidates = documents[:5]
        tasks = [
            self._check_relevance_single(query, doc, i)
            for i, doc in enumerate(initial_candidates)
        ]
        
        results = await asyncio.gather(*tasks)
        reranked_docs = [doc for doc in results if doc is not None]
        reranked_docs.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        
        if not reranked_docs:
            logger.info("All documents filtered out by rerank. Fallback to vector order.")
            return self._fallback_docs(documents, top_k)
            
        return reranked_docs[:top_k]

    def _fallback_docs(self, documents: List[Dict], top_k: int) -> List[Dict]:
        fallback_docs = []
        for doc in documents[:top_k]:
            d = doc.copy()
            if 'parent_content' in d.get('metadata', {}):
                d['content'] = d['metadata']['parent_content']
            fallback_docs.append(d)
        return fallback_docs

    # ----------------------------------------------------------------
    # Step 3 & 4: LitM対策 / 多様性フィルタ
    # ----------------------------------------------------------------
    def reorder_litm(self, documents: List[Dict]) -> List[Dict]:
        if not documents: return []
        dq = deque(documents)
        reordered = []
        if dq: reordered.append(dq.popleft())
        temp_tail = []
        while dq:
            temp_tail.append(dq.popleft())
            if dq:
                reordered.append(dq.popleft())
        return reordered + temp_tail[::-1]

    def filter_diversity(self, documents: List[Dict], threshold: float = 0.7) -> List[Dict]:
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
    # 統合検索メソッド (★修正済み)
    # ----------------------------------------------------------------
    @traceable(name="Search_Pipeline", run_type="chain")
    async def search(
        self, 
        query: str, 
        session_id: str, 
        collection_name: str, 
        top_k: int = 5,
        # ★以下の引数を追加してエラーを回避
        embedding_model: str = "models/text-embedding-004",
        hybrid_weight: float = 0.5,
        **kwargs
    ) -> Dict[str, Any]:
        
        try:
            # 1. クエリ拡張
            expanded_query = await self.expand_query(query)
            logger.info(f"Expanded Query: {expanded_query}")

            # 2. Embedding生成 (渡されたモデルがあればそれを使う)
            query_embedding = await self.get_embedding(expanded_query, model=embedding_model)
            if not query_embedding:
                logger.error("Failed to generate embedding.")
                return {"documents": []}

            # 3. DB検索 (ハイブリッド)
            if not db_client.client:
                logger.error("Database client is not initialized.")
                return {"documents": []}

            match_count = 10
            params = {
                "p_query_text": expanded_query,
                "p_query_embedding": query_embedding,
                "p_match_count": match_count,
                "p_collection_name": collection_name
            }

            try:
                response = db_client.client.rpc("match_documents_hybrid", params).execute()
                documents = response.data
            except Exception as e:
                logger.warning(f"Hybrid search failed ({e}). Fallback to vector search.")
                vector_params = {
                    "p_query_embedding": query_embedding,
                    "p_match_count": match_count,
                    "p_collection_name": collection_name
                }
                response = db_client.client.rpc("match_documents", vector_params).execute()
                documents = response.data

            if not documents:
                return {"documents": []}

            # 4. Rerank (並列実行)
            reranked_docs = await self.rerank(query, documents, top_k)

            # 5. LitM Reorder
            reordered_docs = self.reorder_litm(reranked_docs)

            # 6. Diversity Filter
            final_docs = self.filter_diversity(reordered_docs, threshold=0.7)

            return {"documents": final_docs}

        except Exception as e:
            logger.error(f"Search pipeline error: {e}", exc_info=True)
            return {"documents": []}