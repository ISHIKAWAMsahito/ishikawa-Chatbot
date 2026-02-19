"""
services/search.py

検索パイプライン
- match_unified_search の RPC パラメータ名を p_ プレフィックス付きに修正
- search_mode による資料/FAQ の分岐検索に対応
- クエリ展開ログをDEBUGレベルに変更（Renderログ削減）
"""
import json
import logging
from typing import List, Dict, Any, Optional, Literal

# LangSmith / Utils
from langsmith import traceable
from core.constants import PARAMS
from services.utils import supabase
from models.schemas import SearchResult, DocumentMetadata
from services.llm import LLMService
from services.prompts import QUERY_EXPANSION, RERANK

logger = logging.getLogger(__name__)

# 検索モードの型定義
SearchMode = Literal["hybrid", "documents", "faq"]


class SearchService:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.supabase = supabase

    @traceable(name="Query_Expansion", run_type="chain")
    async def expand_query(self, query: str) -> str:
        """ユーザーのクエリを検索用に拡張します。"""
        if len(query) < 3:
            return query

        prompt = QUERY_EXPANSION.format(query=query)
        try:
            response_stream = await self.llm_service.generate_stream(prompt)
            expanded_text = ""
            async for chunk in response_stream:
                text = chunk.text if hasattr(chunk, "text") else str(chunk)
                expanded_text += text

            cleaned_text = expanded_text.strip()
            # ★ログレベルをDEBUGに変更（Renderログ削減）
            logger.debug(f"Query expansion completed (length: {len(cleaned_text)} chars)")
            return cleaned_text
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return query

    @traceable(name="Rerank_Documents", run_type="chain")
    async def rerank(
        self, query: str, documents: List[SearchResult], top_k: int
    ) -> List[SearchResult]:
        """検索結果をLLMを用いて再評価（リランク）します。"""
        if not documents:
            return []

        candidates_text = ""
        for i, doc in enumerate(documents):
            content_preview = (
                doc.content[:400] + "..." if len(doc.content) > 400 else doc.content
            )
            meta_info = f"[Source: {doc.metadata.source}]"
            candidates_text += f"ID: {i}\nContent: {content_preview}\n{meta_info}\n\n"

        prompt = RERANK.format(
            query=query, count=len(documents) - 1, candidates_text=candidates_text
        )

        try:
            response = await self.llm_service.generate_json(prompt, None)

            try:
                result_text = response.text
                result_json = json.loads(result_text)
                ranked_items = result_json.get("ranked_items", [])
            except Exception:
                logger.warning("Rerank JSON parse failed, returning original order.")
                return documents[:top_k]

            reranked_docs = []
            for item in ranked_items:
                idx = item.get("id")
                score = item.get("score", 0.0)
                reason = item.get("reason", "")
                if idx is not None and 0 <= idx < len(documents):
                    doc = documents[idx]
                    doc.score = score
                    doc.metadata.rerank_reason = reason
                    reranked_docs.append(doc)

            reranked_docs.sort(key=lambda x: x.score, reverse=True)
            return reranked_docs[:top_k]

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents[:top_k]

    def filter_diversity(
        self, documents: List[SearchResult], similarity_threshold: float = 0.85
    ) -> List[SearchResult]:
        """内容が重複しているドキュメントを除外します。"""
        unique_docs = []
        seen_content = set()
        for doc in documents:
            content_sig = doc.content[:50].strip()
            if content_sig in seen_content:
                continue
            seen_content.add(content_sig)
            unique_docs.append(doc)
        return unique_docs

    def reorder_litm(self, documents: List[SearchResult]) -> List[SearchResult]:
        """Lost In The Middle 対策: 重要なドキュメントを先頭と末尾に配置します。"""
        if len(documents) <= 2:
            return documents

        reordered = []
        working_list = documents.copy()
        if working_list:
            reordered.append(working_list.pop(0))

        left = []
        right = []
        for i, doc in enumerate(working_list):
            if i % 2 == 0:
                right.append(doc)
            else:
                left.append(doc)

        return reordered + left + right[::-1]

    # ------------------------------------------------------------------
    # 資料検索（documentsテーブル）
    # ------------------------------------------------------------------
    async def _search_documents(
        self,
        query_embedding: List[float],
        collection_name: str,
        match_count: int = 10,
    ) -> List[SearchResult]:
        """
        一般資料を検索します（match_unified_search RPC）。
        ★ パラメータ名を p_ プレフィックスに修正（PGRST202エラー解消）
        """
        try:
            params_doc = {
                "p_query_embedding": query_embedding,   # ★修正: query_embedding → p_query_embedding
                "p_match_threshold": 0.3,               # ★修正: match_threshold → p_match_threshold
                "p_match_count": match_count,            # ★修正: match_count → p_match_count
                # collection_name は RPC シグネチャに存在しないため除去
            }
            response_docs = self.supabase.rpc("match_unified_search", params_doc).execute()
            docs_data = response_docs.data if response_docs.data else []

            results = []
            for item in docs_data:
                meta = item.get("metadata", {}) or {}
                results.append(
                    SearchResult(
                        id=str(item.get("id")),
                        content=item.get("content", ""),
                        metadata=DocumentMetadata(
                            source=meta.get("source", "不明な資料"),
                            page=meta.get("page"),
                            chunk=meta.get("chunk"),
                            file_path=meta.get("file_path"),
                            url=meta.get("url"),
                            category=meta.get("category"),
                        ),
                        similarity=item.get("similarity", 0.0),
                    )
                )
            logger.debug(f"Document search: {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Document search error: {e}")
            return []

    # ------------------------------------------------------------------
    # FAQ検索（category_fallbacksテーブル）
    # ------------------------------------------------------------------
    async def _search_faq(
        self,
        query_embedding: List[float],
        match_count: int = 5,
    ) -> List[SearchResult]:
        """FAQ（固定回答）を検索します（match_category_fallbacks RPC）。"""
        try:
            params_faq = {
                "p_query_embedding": query_embedding,
                "p_match_threshold": 0.5,
                "p_match_count": match_count,
            }
            response_faq = self.supabase.rpc(
                "match_category_fallbacks", params_faq
            ).execute()
            faq_data = response_faq.data if response_faq.data else []

            results = []
            for item in faq_data:
                raw_meta = item.get("metadata") or {}
                category_val = raw_meta.get("category") or item.get("category_name", "一般FAQ")
                results.append(
                    SearchResult(
                        id=f"faq_{item.get('id')}",
                        content=item.get("content", ""),
                        metadata=DocumentMetadata(
                            source="FAQ",  # ★ FAQフラグ
                            category=category_val,
                            url=raw_meta.get("url"),
                        ),
                        similarity=item.get("similarity", 0.0),
                    )
                )
            logger.debug(f"FAQ search: {len(results)} results")
            return results

        except Exception as e:
            logger.warning(f"FAQ search error (skippable): {e}")
            return []

    # ------------------------------------------------------------------
    # 統合検索パイプライン
    # ------------------------------------------------------------------
    @traceable(name="Search_Pipeline", run_type="chain")
    async def search(
        self,
        query: str,
        session_id: str,
        collection_name: str,
        top_k: int = 5,
        embedding_model: str = "models/gemini-embedding-001",
        search_mode: SearchMode = "hybrid",
    ) -> Dict[str, Any]:
        """
        統合検索パイプライン

        search_mode:
            "hybrid"    - 資料 + FAQ を同時検索してマージ（デフォルト）
            "documents" - 一般資料のみ検索
            "faq"       - FAQ のみ検索
        """
        # 1. 埋め込みベクトル生成
        query_embedding = await self.llm_service.get_embedding(
            query, model=embedding_model
        )
        if not query_embedding:
            return {"documents": [], "is_faq_match": False}

        initial_candidates: List[SearchResult] = []

        # 2. search_mode に応じて検索を実行
        if search_mode in ("hybrid", "documents"):
            doc_results = await self._search_documents(
                query_embedding, collection_name, match_count=10
            )
            initial_candidates.extend(doc_results)

        if search_mode in ("hybrid", "faq"):
            faq_results = await self._search_faq(query_embedding, match_count=5)
            initial_candidates.extend(faq_results)

        if not initial_candidates:
            logger.info(f"No candidates found (mode={search_mode})")
            return {"documents": [], "is_faq_match": False}

        # 3. リランク
        rerank_input = initial_candidates[:15]
        reranked_results = await self.rerank(query, rerank_input, top_k)

        if not reranked_results:
            return {"documents": [], "is_faq_match": False}

        # 4. FAQモード判定（AND条件）
        top_doc = reranked_results[0]
        rerank_threshold = PARAMS.get("RERANK_SCORE_THRESHOLD", 6.0)
        similarity_threshold = PARAMS.get("QA_SIMILARITY_THRESHOLD", 0.90)

        is_faq_source = top_doc.metadata.source == "FAQ"
        is_high_score = top_doc.score >= rerank_threshold
        is_high_similarity = top_doc.similarity >= similarity_threshold

        if is_faq_source and is_high_score and is_high_similarity:
            # --- FAQ一致モード ---
            logger.info(
                f"FAQ Match Triggered: Score={top_doc.score:.1f}, Sim={top_doc.similarity:.3f}"
            )
            return {
                "documents": [top_doc.model_dump()],
                "is_faq_match": True,
            }
        else:
            # --- 通常資料モード ---
            filtered_docs = self.filter_diversity(reranked_results)
            final_documents = self.reorder_litm(filtered_docs)
            logger.info(
                f"Standard Document Mode (mode={search_mode}, docs={len(final_documents)})"
            )
            return {
                "documents": [doc.model_dump() for doc in final_documents],
                "is_faq_match": False,
            }