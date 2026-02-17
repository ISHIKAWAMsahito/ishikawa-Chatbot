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
from core.constants import PARAMS

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------
# URL生成ロジック (サービスの有無に関わらず動作するよう強化)
# -----------------------------------------------------------------
try:
    # まず正規のストレージサービスからのインポートを試みる
    from services.storage import generate_signed_url
except ImportError:
    # モジュールがない場合は、DBクライアントを使ってここで直接生成する (フォールバック)
    logger.warning("services.storage not found. Using fallback URL generation.")
    
    async def generate_signed_url(path: str) -> str:
        try:
            if not path: return ""
            # 環境変数からバケット名を取得 (デフォルト: slides)
            bucket_name = os.getenv("SUPABASE_STORAGE_BUCKET", "slides")
            
            # DBクライアント経由で署名付きURLを発行 (有効期限: 3600秒)
            # db_client.client は supabase.Client を指すと仮定
            res = db_client.client.storage.from_(bucket_name).create_signed_url(path, 3600)
            
            # レスポンスが辞書か文字列かで分岐（ライブラリのバージョン差異対策）
            if isinstance(res, dict):
                return res.get("signedURL", "")
            elif isinstance(res, str):
                return res
            return ""
        except Exception as e:
            logger.error(f"Fallback URL generation failed for '{path}': {e}")
            return ""

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

class RerankItem(BaseModel):
    id: int = Field(..., description="ドキュメントのインデックス番号")
    score: float = Field(..., description="関連性スコア (0.0-10.0)")
    reason: str = Field(..., description="採点理由")

class BatchRerankResult(BaseModel):
    ranked_items: List[RerankItem]

# =================================================================
# 2. 検索サービスの定義
# =================================================================

class SearchService:
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    async def get_embedding(self, text: str, model: str = EMBEDDING_MODEL_DEFAULT) -> List[float]:
        try:
            return await self.llm.get_embedding(text, model)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}", exc_info=True)
            return []

    @alru_cache(maxsize=100)
    @traceable(name="Step1_Query_Expansion", run_type="chain")
    async def expand_query(self, query: str) -> str:
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

    @traceable(name="Step2_AI_Rerank_Batch", run_type="chain")
    async def rerank(self, query: str, documents: List[SearchResult], top_k: int) -> List[SearchResult]:
        if not documents:
            return []

        rerank_input_count = PARAMS.get("RERANK_TOP_K_INPUT", 5)
        candidates = documents[:rerank_input_count]

        candidates_text = ""
        for i, doc in enumerate(candidates):
            eval_content = doc.metadata.parent_content or doc.content
            content_preview = eval_content[:800].replace('\n', ' ')
            candidates_text += f"ID: {i}\nContent: {content_preview}\n\n"

        prompt = prompts.RERANK.format(
            query=query,
            count=len(candidates) - 1,
            candidates_text=candidates_text
        )

        try:
            response = await self.llm.generate_json(prompt, BatchRerankResult)
            
            if hasattr(response, 'text'):
                text = response.text
            else:
                text = str(response)
            
            text = text.replace("```json", "").replace("```", "").strip()
            data = json.loads(text)
            result_obj = BatchRerankResult(**data)
            
            reranked_docs = []
            threshold = PARAMS.get("RERANK_SCORE_THRESHOLD", 6.0)

            for item in result_obj.ranked_items:
                if 0 <= item.id < len(candidates):
                    doc = candidates[item.id]
                    if item.score >= threshold:
                        doc.metadata.rerank_score = item.score
                        doc.metadata.rerank_reason = item.reason
                        
                        if doc.metadata.parent_content:
                            doc.content = doc.metadata.parent_content
                        
                        reranked_docs.append(doc)

            reranked_docs.sort(key=lambda x: x.metadata.rerank_score, reverse=True)
            
            if not reranked_docs:
                logger.info("Rerank filtered all docs. Fallback to similarity order.")
                fallback_docs = documents[:top_k]
                for d in fallback_docs:
                    if d.metadata.parent_content:
                        d.content = d.metadata.parent_content
                return fallback_docs

            return reranked_docs[:top_k]

        except Exception as e:
            logger.error(f"Batch rerank failed: {e}", exc_info=True)
            return documents[:top_k]

    def reorder_litm(self, documents: List[SearchResult]) -> List[SearchResult]:
        if not documents: return []
        dq = deque(documents)
        reordered = []
        while dq:
            reordered.append(dq.popleft())
            if dq:
                reordered.insert(len(reordered)-1, dq.popleft())
        return reordered

    def filter_diversity(self, documents: List[SearchResult], threshold: float = 0.7) -> List[SearchResult]:
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
        """署名付きURL付与（強化版）"""
        logger.info(f"Starting URL enrichment for {len(documents)} documents.")
        
        for i, doc in enumerate(documents):
            source_path = doc.metadata.source
            
            # --- 修正: FAQやソース不明なものはスキップ ---
            if not source_path or source_path == "不明な資料":
                continue
            # FAQデータはファイル実体がないのでスキップ
            if source_path == "FAQ" or source_path.startswith("FAQ"):
                continue
                
            if doc.metadata.url: continue

            try:
                # URL生成実行
                signed_url = await generate_signed_url(source_path)
                if signed_url:
                    doc.metadata.url = signed_url
                else:
                    logger.debug(f"[Doc {i}] URL generation returned empty for source: {source_path}")
            except Exception as e:
                logger.warning(f"[Doc {i}] Error generating URL for '{source_path}': {e}")
                
        return documents

    @traceable(name="Search_Pipeline", run_type="chain")
    async def search(
        self, 
        query: str, 
        session_id: str, 
        collection_name: str, 
        top_k: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """統合検索パイプライン (Unified Search: Documents + FAQ)"""
        try:
            # 1. クエリ拡張
            expanded_query = await self.expand_query(query)

            # 2. Embedding生成
            query_embedding = await self.get_embedding(expanded_query)
            if not query_embedding: return {"documents": [], "is_faq_match": False}

            # 3. DB検索 (Unified Search)
            # リランク前の取得候補数
            match_count = PARAMS.get("RERANK_TOP_K_INPUT", 15)
            # 足切りライン（低すぎるスコアのものはDB側で除外）
            similarity_threshold = 0.3 
            
            # SQL関数呼び出し
            # NOTE: 前手順でSQL関数の戻り値型を text に変更済みであること
            response = db_client.client.rpc("match_unified_search", {
                "p_query_embedding": query_embedding,
                "p_match_threshold": similarity_threshold,
                "p_match_count": match_count
            }).execute()
            
            # データの受け取り処理
            raw_docs = []
            if response and hasattr(response, 'data'):
                for d in response.data:
                    meta_data = d.get('metadata', {})
                    
                    # メタデータのパース処理強化（DBから文字列で来た場合に対応）
                    if isinstance(meta_data, str):
                        try:
                            meta_data = json.loads(meta_data)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse metadata JSON: {meta_data[:50]}...")
                            meta_data = {}
                    
                    if not isinstance(meta_data, dict):
                        meta_data = {}

                    raw_docs.append(
                        SearchResult(
                            id=d.get('id'), # IDは文字列(uuid/text)または数値(int)に対応
                            content=d.get('content'),
                            metadata=DocumentMetadata(**meta_data),
                            similarity=d.get('similarity', 0.0)
                        )
                    )

            # 4. パイプライン実行（バッチリランク呼び出し）
            reranked = await self.rerank(query, raw_docs, top_k)
            
            # FAQ一致判定ロジック
            is_faq_match = False
            if reranked:
                top_doc = reranked[0]
                
                sim_threshold = PARAMS.get("QA_SIMILARITY_THRESHOLD", 0.90)
                score_threshold = PARAMS.get("RERANK_SCORE_THRESHOLD", 6.0)
                
                # FAQテーブル由来 または 高スコア判定
                is_faq_source = (top_doc.metadata.source == "FAQ")
                high_score = (top_doc.similarity >= sim_threshold and 
                              top_doc.metadata.rerank_score >= score_threshold)
                
                # FAQ由来なら閾値超えで即マッチとみなす
                if (is_faq_source and top_doc.metadata.rerank_score >= score_threshold) or high_score:
                    is_faq_match = True
                    logger.info(f"[Search] FAQ/High-Confidence match. ID: {top_doc.id}, Source: {top_doc.metadata.source}")

            reordered = self.reorder_litm(reranked)
            final_docs = self.filter_diversity(reordered)
            
            # 5. URL付与 (修正版呼び出し)
            final_docs_with_url = await self._enrich_with_urls(final_docs)

            return {
                "documents": [d.model_dump() for d in final_docs_with_url],
                "is_faq_match": is_faq_match
            }

        except Exception as e:
            logger.error(f"Search pipeline error: {e}", exc_info=True)
            return {"documents": [], "is_faq_match": False}