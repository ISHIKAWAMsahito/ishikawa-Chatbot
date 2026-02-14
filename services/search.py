import json
import logging
import asyncio
import os
from typing import List, Dict, Any, Optional, Union
from difflib import SequenceMatcher
from collections import deque

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, AsyncRetrying
from async_lru import alru_cache
from langsmith import traceable
from pydantic import BaseModel, Field

# 内部モジュールのインポート
from services.llm import LLMService
from services import prompts
from core.database import db_client
from core.config import GEMINI_API_KEY, EMBEDDING_MODEL_DEFAULT
from core.constants import PARAMS  # 【追加】定数読み込み

# Storageサービスのインポート (エラー回避付き)
try:
    from services.storage import generate_signed_url
except ImportError:
    async def generate_signed_url(path: str) -> str:
        return ""

logger = logging.getLogger(__name__)

# Gemini API設定
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# =================================================================
# 1. データモデルの定義
# =================================================================

class DocumentMetadata(BaseModel):
    source: str = "不明な資料"
    url: Optional[str] = None
    parent_content: Optional[str] = None
    page: Optional[int] = None
    rerank_score: float = 0.0
    rerank_reason: str = ""
    model_config = {"extra": "ignore"}

class SearchResult(BaseModel):
    id: Union[str, int]
    content: str
    metadata: DocumentMetadata
    similarity: float = 0.0
    model_config = {"extra": "ignore"}

# =================================================================
# 2. 検索サービスの定義
# =================================================================

class SearchService:
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    async def get_embedding(self, text: str, model: str = EMBEDDING_MODEL_DEFAULT) -> List[float]:
        """テキストをベクトル化する"""
        try:
            return await self.llm.get_embedding(text, model)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}", exc_info=True)
            return []

    @alru_cache(maxsize=100)
    @traceable(name="Step1_Query_Expansion", run_type="chain")
    async def expand_query(self, query: str) -> str:
        """クエリ拡張"""
        cot_prompt = prompts.QUERY_EXPANSION.format(query=query)
        try:
            response = await self.llm.generate_stream(cot_prompt)
            full_text = ""
            async for chunk in response:
                if chunk.text: full_text += chunk.text
            
            expanded = full_text.strip()
            return expanded if expanded else query
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return query

    async def _check_relevance_single(self, query: str, doc: SearchResult, index: int) -> Optional[SearchResult]:
        """単一ドキュメントの並列評価 (Re-ranking)"""
        eval_content = doc.metadata.parent_content or doc.content
        snippet = eval_content[:500].replace('\n', ' ')
        
        prompt = f"""
        あなたは検索エンジンのリランクシステムです。
        以下の質問に対するドキュメントの関連性を0.0〜10.0で評価し、JSONで答えてください。
        質問: {query}
        ドキュメント: {snippet}
        出力形式: {{"score": 8.5, "reason": "理由"}}
        """

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(2), 
                wait=wait_exponential(min=1, max=3),
                retry=retry_if_exception_type(Exception)
            ):
                with attempt:
                    model = genai.GenerativeModel("models/gemini-2.5-flash")
                    response = await model.generate_content_async(
                        prompt,
                        generation_config={"response_mime_type": "application/json"}
                    )
                    
                    text = response.text.replace("```json", "").replace("```", "").strip()
                    data = json.loads(text)
                    score = float(data.get("score", 0.0))
                    
                    # 【修正】定数を使用してフィルタリング
                    threshold = PARAMS.get("RERANK_SCORE_THRESHOLD", 6.0)
                    if score >= threshold:
                        doc.metadata.rerank_score = score
                        doc.metadata.rerank_reason = data.get("reason", "")
                        if doc.metadata.parent_content:
                            doc.content = doc.metadata.parent_content
                        return doc
                    return None
        except Exception as e:
            logger.error(f"Relevance check failed for doc {index}: {e}")
            return None

    @traceable(name="Step2_AI_Rerank", run_type="chain")
    async def rerank(self, query: str, documents: List[SearchResult], top_k: int) -> List[SearchResult]:
        """AIリランク"""
        if not documents:
            return []

        candidates = documents[:5]
        tasks = [self._check_relevance_single(query, doc, i) for i, doc in enumerate(candidates)]
        
        results = await asyncio.gather(*tasks)
        reranked = [doc for doc in results if doc is not None]
        
        reranked.sort(key=lambda x: x.metadata.rerank_score, reverse=True)
        
        if not reranked:
            logger.info("Rerank filtered all docs. Fallback to similarity order.")
            return documents[:top_k]
            
        return reranked[:top_k]

    def reorder_litm(self, documents: List[SearchResult]) -> List[SearchResult]:
        """Lost in the Middle対策"""
        if not documents: return []
        dq = deque(documents)
        reordered = []
        while dq:
            reordered.append(dq.popleft())
            if dq:
                reordered.insert(len(reordered)-1, dq.popleft())
        return reordered

    def filter_diversity(self, documents: List[SearchResult], threshold: float = 0.7) -> List[SearchResult]:
        """多様性フィルタ"""
        unique_docs = []
        for doc in documents:
            is_duplicate = False
            for selected in unique_docs:
                sim = SequenceMatcher(None, doc.content, selected.content).ratio()
                if sim > threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_docs.append(doc)
        return unique_docs

    async def _enrich_with_urls(self, documents: List[SearchResult]) -> List[SearchResult]:
        """署名付きURL付与"""
        logger.info(f"Starting URL enrichment for {len(documents)} documents.")
        for i, doc in enumerate(documents):
            source_path = doc.metadata.source
            if not source_path: continue
            if doc.metadata.url: continue

            try:
                signed_url = await generate_signed_url(source_path)
                if signed_url:
                    doc.metadata.url = signed_url
            except Exception as e:
                logger.error(f"[Doc {i}] Error generating URL: {e}", exc_info=True)
        return documents

    @traceable(name="Search_Pipeline", run_type="chain")
    async def search(
        self, 
        query: str, 
        session_id: str, 
        collection_name: str, 
        top_k: int = 5,
        **kwargs
    ) -> Dict[str, Any]: # 【修正】戻り値をDict[str, Any]に変更
        """統合検索パイプライン"""
        try:
            # 1. クエリ拡張
            expanded_query = await self.expand_query(query)

            # 2. Embedding生成
            query_embedding = await self.get_embedding(expanded_query)
            if not query_embedding: return {"documents": [], "is_faq_match": False}

            # 3. DB検索
            # 【修正】定数を使用して検索数を制御
            match_count = PARAMS.get("RERANK_TOP_K_INPUT", 15)
            response = db_client.client.rpc("match_documents_hybrid", {
                "p_query_text": expanded_query,
                "p_query_embedding": query_embedding,
                "p_match_count": match_count,
                "p_collection_name": collection_name
            }).execute()
            
            raw_docs = []
            for d in response.data:
                meta_data = d.get('metadata', {})
                if isinstance(meta_data, str):
                    try:
                        meta_data = json.loads(meta_data)
                    except:
                        meta_data = {}
                
                raw_docs.append(
                    SearchResult(
                        id=d.get('id'),
                        content=d.get('content'),
                        metadata=DocumentMetadata(**meta_data),
                        similarity=d.get('similarity', 0.0)
                    )
                )

            # 4. パイプライン実行
            reranked = await self.rerank(query, raw_docs, top_k)
            
            # 【追加】FAQ一致判定ロジック
            is_faq_match = False
            if reranked:
                top_doc = reranked[0]
                # 類似度0.9以上 かつ リランクスコア基準以上
                sim_threshold = PARAMS.get("QA_SIMILARITY_THRESHOLD", 0.90)
                score_threshold = PARAMS.get("RERANK_SCORE_THRESHOLD", 6.0)
                
                if (top_doc.similarity >= sim_threshold and 
                    top_doc.metadata.rerank_score >= score_threshold):
                    is_faq_match = True
                    logger.info(f"[Search] High confidence FAQ match detected. ID: {top_doc.id}")

            reordered = self.reorder_litm(reranked)
            final_docs = self.filter_diversity(reordered)
            
            # 5. URL付与
            final_docs_with_url = await self._enrich_with_urls(final_docs)

            return {
                "documents": [d.model_dump() for d in final_docs_with_url],
                "is_faq_match": is_faq_match # フラグを追加
            }

        except Exception as e:
            logger.error(f"Search pipeline error: {e}", exc_info=True)
            return {"documents": [], "is_faq_match": False}