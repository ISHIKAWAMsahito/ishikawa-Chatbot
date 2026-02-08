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
# 1. データモデルの定義 (ここがないとエラーになります)
# =================================================================

class DocumentMetadata(BaseModel):
    source: str = "不明な資料"
    url: Optional[str] = None
    parent_content: Optional[str] = None
    page: Optional[int] = None
    rerank_score: float = 0.0
    rerank_reason: str = ""
    # 予期せぬフィールドが来ても許容する設定
    model_config = {"extra": "ignore"}

class SearchResult(BaseModel):
    id: Union[str, int]
    content: str
    metadata: DocumentMetadata
    similarity: float = 0.0
    # 予期せぬフィールドが来ても許容する設定
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
        """クエリ拡張 (Fail Fastではなく元のクエリを返す防御的設計)"""
        cot_prompt = prompts.QUERY_EXPANSION.format(query=query)
        try:
            # 短いタイムアウトで試行
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
        # 親コンテンツがあれば優先的に評価対象にする
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
            # 軽量モデル(Flash)を使用し、並列リクエストのコストを下げる
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(2), 
                wait=wait_exponential(min=1, max=3),
                retry=retry_if_exception_type(Exception)
            ):
                with attempt:
                    model = genai.GenerativeModel("gemini-2.5-flash")
                    response = await model.generate_content_async(
                        prompt,
                        generation_config={"response_mime_type": "application/json"}
                    )
                    
                    text = response.text.replace("```json", "").replace("```", "").strip()
                    data = json.loads(text)
                    score = float(data.get("score", 0.0))
                    
                    # 標準仕様: 6.0点以上のものだけを採用
                    if score >= 6.0:
                        doc.metadata.rerank_score = score
                        doc.metadata.rerank_reason = data.get("reason", "")
                        # 親コンテンツがある場合は本文を親のものに差し替えて情報量を増やす
                        if doc.metadata.parent_content:
                            doc.content = doc.metadata.parent_content
                        return doc
                    return None
        except Exception as e:
            logger.error(f"Relevance check failed for doc {index}: {e}")
            return None

    @traceable(name="Step2_AI_Rerank", run_type="chain")
    async def rerank(self, query: str, documents: List[SearchResult], top_k: int) -> List[SearchResult]:
        """標準仕様に基づいたAIリランク (上位5件精査)"""
        if not documents:
            return []

        # 仕様: 上位5件のみを精査対象とする
        candidates = documents[:5]
        tasks = [self._check_relevance_single(query, doc, i) for i, doc in enumerate(candidates)]
        
        # 並列実行 (Gather)
        results = await asyncio.gather(*tasks)
        reranked = [doc for doc in results if doc is not None]
        
        # スコア順にソート
        reranked.sort(key=lambda x: x.metadata.rerank_score, reverse=True)
        
        if not reranked:
            logger.info("Rerank filtered all docs. Fallback to similarity order.")
            return documents[:top_k]
            
        return reranked[:top_k]

    def reorder_litm(self, documents: List[SearchResult]) -> List[SearchResult]:
        """標準仕様: LitM配置 (U字型)"""
        if not documents: return []
        dq = deque(documents)
        reordered = []
        
        # 1位を最初、2位を最後、3位を最初...の順で配置
        while dq:
            reordered.append(dq.popleft()) # 最初へ
            if dq:
                reordered.insert(len(reordered)-1, dq.popleft()) # 末尾寄りへ
        
        return reordered

    def filter_diversity(self, documents: List[SearchResult], threshold: float = 0.7) -> List[SearchResult]:
        """標準仕様: 多様性フィルタ (70%重複カット)"""
        unique_docs = []
        for doc in documents:
            is_duplicate = False
            for selected in unique_docs:
                # 文字列の類似度を確認
                sim = SequenceMatcher(None, doc.content, selected.content).ratio()
                if sim > threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_docs.append(doc)
        return unique_docs

    async def _enrich_with_urls(self, documents: List[SearchResult]) -> List[SearchResult]:
        """
        検索結果のドキュメントに署名付きURLを付与する。
        詳細なログを出力し、トラブルシューティングを容易にする。
        """
        logger.info(f"Starting URL enrichment for {len(documents)} documents.")
        
        for i, doc in enumerate(documents):
            # Pydanticモデルなのでドット記法でアクセス
            source_path = doc.metadata.source
            
            # 1. sourceパスの有無を確認
            if not source_path:
                logger.warning(f"[Doc {i}] 'source' metadata is missing or empty. Skipping URL generation.")
                continue

            # 2. 既にURLがある場合はスキップ（二重生成防止）
            if doc.metadata.url:
                logger.debug(f"[Doc {i}] URL already exists for {source_path}. Skipping.")
                continue

            try:
                # 3. URL生成の試行
                logger.debug(f"[Doc {i}] Generating signed URL for: {source_path}")
                
                # generate_signed_url は services.storage からインポートされている前提
                signed_url = await generate_signed_url(source_path)
                
                if signed_url:
                    doc.metadata.url = signed_url
                    logger.info(f"[Doc {i}] ✅ Successfully generated URL for: {source_path}")
                else:
                    logger.warning(f"[Doc {i}] ⚠️ URL generation returned empty for: {source_path}")
                    
            except Exception as e:
                logger.error(f"[Doc {i}] ❌ Error generating URL for {source_path}: {e}", exc_info=True)
                # エラー時はURLなしで続行（検索結果自体は返す）

        return documents

    @traceable(name="Search_Pipeline", run_type="chain")
    async def search(
        self, 
        query: str, 
        session_id: str, 
        collection_name: str, 
        top_k: int = 5,
        **kwargs
    ) -> Dict[str, List[Dict[str, Any]]]:
        """統合検索パイプライン"""
        try:
            # 1. クエリ拡張
            expanded_query = await self.expand_query(query)

            # 2. Embedding生成
            query_embedding = await self.get_embedding(expanded_query)
            if not query_embedding: return {"documents": []}

            # 3. DB検索
            response = db_client.client.rpc("match_documents_hybrid", {
                "p_query_text": expanded_query,
                "p_query_embedding": query_embedding,
                "p_match_count": 10,
                "p_collection_name": collection_name
            }).execute()
            
            # 生データをPydanticモデルに変換 (Strict Typing)
            raw_docs = []
            for d in response.data:
                # metadataが辞書であることを確認
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

            # 4. パイプライン実行 (Rerank -> LitM -> Diversity)
            reranked = await self.rerank(query, raw_docs, top_k)
            reordered = self.reorder_litm(reranked)
            final_docs = self.filter_diversity(reordered)
            
            # 5. URL付与 (ログ出力付き)
            final_docs_with_url = await self._enrich_with_urls(final_docs)

            # 辞書として返す (CamelCase変換はFastAPI側またはモデル設定で対応)
            return {"documents": [d.model_dump() for d in final_docs_with_url]}

        except Exception as e:
            logger.error(f"Search pipeline error: {e}", exc_info=True)
            return {"documents": []}