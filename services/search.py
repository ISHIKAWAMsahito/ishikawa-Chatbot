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

# ★追加: バッチリランク解析用モデル
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

    @traceable(name="Step2_AI_Rerank_Batch", run_type="chain")
    async def rerank(self, query: str, documents: List[SearchResult], top_k: int) -> List[SearchResult]:
        """
        AIリランク（バッチ処理版）
        上位候補をまとめてLLMに渡し、一括でスコアリングを行うことでAPI回数を1回に削減。
        """
        if not documents:
            return []

        # 1. リランク対象の選定
        # 全件やるとトークン過多になるため、PARAMSで指定された上位件数(デフォルト5)のみリランク
        rerank_input_count = PARAMS.get("RERANK_TOP_K_INPUT", 5)
        candidates = documents[:rerank_input_count]

        # 2. プロンプト用のコンテキスト作成
        # トークン節約のため、各ドキュメントは先頭800文字程度に制限
        candidates_text = ""
        for i, doc in enumerate(candidates):
            # 親コンテンツがある場合はそちらを優先して評価に使う
            eval_content = doc.metadata.parent_content or doc.content
            content_preview = eval_content[:800].replace('\n', ' ')
            candidates_text += f"ID: {i}\nContent: {content_preview}\n\n"

        prompt = prompts.RERANK.format(
            query=query,
            count=len(candidates) - 1,
            candidates_text=candidates_text
        )

        try:
            # 3. LLMによる一括判定 (JSONモード)
            # llm.generate_json は response_schema に対応している前提
            response = await self.llm.generate_json(prompt, BatchRerankResult)
            
            # レスポンスのテキスト取得
            if hasattr(response, 'text'):
                text = response.text
            else:
                text = str(response)
            
            # JSONクリーニング
            text = text.replace("```json", "").replace("```", "").strip()
            # 万が一JSON以外の文字が混じっている場合の簡易ガードは本来必要だが、
            # GeminiのJSONモードは信頼性が高いためここでは直接パース
            data = json.loads(text)
            
            # Pydanticで検証
            result_obj = BatchRerankResult(**data)
            
            # 4. スコアのマッピングとフィルタリング
            reranked_docs = []
            threshold = PARAMS.get("RERANK_SCORE_THRESHOLD", 6.0)

            for item in result_obj.ranked_items:
                # IDが範囲内かチェック
                if 0 <= item.id < len(candidates):
                    doc = candidates[item.id]
                    
                    # スコア基準チェック
                    if item.score >= threshold:
                        doc.metadata.rerank_score = item.score
                        doc.metadata.rerank_reason = item.reason
                        
                        # リランク合格時、回答生成用に「親チャンク」へ内容を差し替え
                        # (検索は小さいチャンクで行い、回答は大きいコンテキストで行うため)
                        if doc.metadata.parent_content:
                            doc.content = doc.metadata.parent_content
                        
                        reranked_docs.append(doc)

            # 5. スコア順にソート
            reranked_docs.sort(key=lambda x: x.metadata.rerank_score, reverse=True)
            
            # 誰も基準を超えなかった場合、検索結果ゼロにするのではなく、
            # 類似度検索の上位をそのまま返す（フォールバック）
            if not reranked_docs:
                logger.info("Rerank filtered all docs. Fallback to similarity order.")
                # ただしこの場合は parent_content への置換が行われないため手動で行う
                fallback_docs = documents[:top_k]
                for d in fallback_docs:
                    if d.metadata.parent_content:
                        d.content = d.metadata.parent_content
                return fallback_docs

            return reranked_docs[:top_k]

        except Exception as e:
            logger.error(f"Batch rerank failed: {e}", exc_info=True)
            # エラー時はFail Safeとして、元のベクトル検索結果をそのまま返す
            return documents[:top_k]

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
    ) -> Dict[str, Any]:
        """統合検索パイプライン (Unified Search: Documents + FAQ)"""
        try:
            # 1. クエリ拡張 (変更なし)
            expanded_query = await self.expand_query(query)

            # 2. Embedding生成 (変更なし)
            query_embedding = await self.get_embedding(expanded_query)
            if not query_embedding: return {"documents": [], "is_faq_match": False}

            # 3. DB検索 (★ここを修正)
            # 以前の match_documents_hybrid から match_unified_search に変更
            
            # リランク前の取得候補数
            match_count = PARAMS.get("RERANK_TOP_K_INPUT", 15)
            # 足切りライン（低すぎるスコアのものはDB側で除外）
            similarity_threshold = 0.3 
            
            # ★ SQLで定義した関数を呼び出す
            response = db_client.client.rpc("match_unified_search", {
                "p_query_embedding": query_embedding,
                "p_match_threshold": similarity_threshold,
                "p_match_count": match_count
            }).execute()
            
            # --- データの受け取り処理 (既存コードとほぼ同じでOK) ---
            raw_docs = []
            for d in response.data:
                meta_data = d.get('metadata', {})
                
                # 文字列で返ってきた場合のパース処理（既存維持）
                if isinstance(meta_data, str):
                    try:
                        meta_data = json.loads(meta_data)
                    except:
                        meta_data = {}
                
                # SQLで 'source_type' を返していますが、
                # metadata内の 'source' (FAQの場合は自動でセット済) が優先されます
                
                raw_docs.append(
                    SearchResult(
                        id=d.get('id'),
                        content=d.get('content'), # FAQの場合はここが回答文になっています
                        metadata=DocumentMetadata(**meta_data),
                        similarity=d.get('similarity', 0.0)
                    )
                )

            # 4. パイプライン実行（バッチリランク呼び出し）(変更なし)
            reranked = await self.rerank(query, raw_docs, top_k)
            
            # FAQ一致判定ロジック (★少し改良)
            is_faq_match = False
            if reranked:
                top_doc = reranked[0]
                
                # 設定値の取得
                sim_threshold = PARAMS.get("QA_SIMILARITY_THRESHOLD", 0.90)
                score_threshold = PARAMS.get("RERANK_SCORE_THRESHOLD", 6.0)
                
                # 条件A: 類似度とリランクスコアが高い（既存ロジック）
                high_score = (top_doc.similarity >= sim_threshold and 
                              top_doc.metadata.rerank_score >= score_threshold)
                              
                # 条件B: データソースがFAQであり、かつリランクスコアが合格点
                # (FAQデータは人間が作った正解なので、信頼度を少し甘く見ても良い場合の判定)
                is_faq_source = top_doc.metadata.source == "FAQ"
                faq_hit = is_faq_source and (top_doc.metadata.rerank_score >= score_threshold)

                if high_score or faq_hit:
                    is_faq_match = True
                    logger.info(f"[Search] FAQ/High-Confidence match detected. ID: {top_doc.id}, Source: {top_doc.metadata.source}")

            reordered = self.reorder_litm(reranked)
            final_docs = self.filter_diversity(reordered)
            
            # 5. URL付与 (変更なし)
            final_docs_with_url = await self._enrich_with_urls(final_docs)

            return {
                "documents": [d.model_dump() for d in final_docs_with_url],
                "is_faq_match": is_faq_match
            }

        except Exception as e:
            logger.error(f"Search pipeline error: {e}", exc_info=True)
            return {"documents": [], "is_faq_match": False}