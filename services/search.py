import json
import logging
import asyncio
from typing import List, Dict, Optional
from difflib import SequenceMatcher
from collections import deque

import typing_extensions as typing
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, AsyncRetrying, RetryError
from async_lru import alru_cache
from langsmith import traceable

from services.llm import LLMService
from services import prompts
from core.constants import PARAMS

logger = logging.getLogger(__name__)

class RankedItem(typing.TypedDict):
    id: int
    score: float
    reason: str

class RerankResponse(typing.TypedDict):
    ranked_items: list[RankedItem]

class SearchService:
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    # ----------------------------------------------------------------
    # Step 1: クエリ拡張
    # ----------------------------------------------------------------
    @alru_cache(maxsize=100)
    @traceable(name="Step1_Query_Expansion", run_type="chain")
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        stop=stop_after_attempt(5), 
        reraise=True 
    )
    async def expand_query(self, query: str) -> str:
        cot_prompt = f"""
        あなたは世界最高峰の検索エンジニアです。以下のユーザーの質問に対して、最適な検索クエリを作成してください。
        【プロセス】
        1. ユーザーの質問の「真の意図」と「不足している前提知識」を分析してください。
        2. 専門用語や同義語、具体的な関連語句をリストアップしてください。
        3. それらを踏まえ、検索エンジンに投げるべき「具体的かつ網羅的な検索クエリ」を生成してください。
        質問: {query}
        出力は最終的な検索クエリ文字列のみを行ってください。余計な説明は不要です。
        """
        try:
            response = await self.llm.generate_stream(cot_prompt)
            full_text = ""
            async for chunk in response:
                if chunk.text: full_text += chunk.text
            return full_text.strip()
        except Exception as e:
            logger.warning(f"Query expansion error: {e}")
            raise e

    # ----------------------------------------------------------------
    # Step 2: AIリランク (親子チャンキング対応版)
    # ----------------------------------------------------------------
    @traceable(name="Step2_AI_Rerank", run_type="chain")
    async def rerank(self, query: str, documents: List[Dict], top_k: int) -> List[Dict]:
        """
        Step 2: リランク
        - 親子チャンキング対応: 親コンテンツ(parent_content)があれば優先して使用
        - 耐障害性: エラー時は元の順序でフォールバック（その際も親コンテンツへの差替は実施）
        """
        if not documents:
            return []

        # 1. リランク用プロンプト作成
        # ここでAIに読ませるテキストを「親チャンク」にする
        candidates_text = ""
        for i, doc in enumerate(documents):
            meta = doc.get('metadata', {})
            
            # ★ 親子チャンキング対応: 親があれば親を使う
            content_for_llm = meta.get('parent_content', doc.get('content', ''))
            
            # 親チャンクは情報量が多いので、文字数制限を少し緩和 (1200 -> 2500)
            snippet = content_for_llm[:2500].replace('\n', ' ') 
            
            candidates_text += f"Document ID: {i}\nSource: {meta.get('source', 'Unknown')}\nContent: {snippet}\n\n---\n\n"

        rerank_prompt = f"""
        あなたは検索ランキングの専門家です。以下のクエリに対して、各ドキュメントの関連性を0.0〜10.0のスコアで厳密に評価してください。
        【クエリ】
        {query}
        【評価基準】
        - 10点: クエリに対する直接的な回答が含まれており、これだけで解決する。
        - 7-9点: 非常に有益な情報を含み、回答の核となる。
        - 4-6点: 関連するトピックだが、直接的な回答ではない、または情報が断片的。
        - 0-3点: クエリと無関係、またはノイズ。
        【候補ドキュメント】
        {candidates_text}
        結果は必ず以下のJSONフォーマットで出力してください。
        {{
            "ranked_items": [
                {{ "id": 0, "score": 9.5, "reason": "..." }},
                ...
            ]
        }}
        """

        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(Exception),
                wait=wait_exponential(multiplier=2, min=2, max=30),
                stop=stop_after_attempt(5), # UXを考慮し、最大5回程度に留める
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
                        
                        if idx is not None and isinstance(idx, int) and 0 <= idx < len(documents):
                            if score >= PARAMS.get("RERANK_SCORE_THRESHOLD", 3.0):
                                doc = documents[idx].copy()
                                
                                # ★ 重要: 回答生成用に、中身を「親チャンク」に差し替える
                                # これにより、最終回答を作るAIが文脈全体を読めるようになる
                                if 'parent_content' in doc.get('metadata', {}):
                                    doc['content'] = doc['metadata']['parent_content']

                                doc['rerank_score'] = score
                                doc['rerank_reason'] = reason
                                reranked_docs.append(doc)

                    reranked_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
                    
                    logger.info(f"Rerank success. Top score: {reranked_docs[0]['rerank_score'] if reranked_docs else 0}")
                    return reranked_docs[:top_k]

        except RetryError:
            logger.error("Rerank failed after retries. Falling back to original order.")
            # フォールバック時も、可能な限り親チャンクに差し替えて返す（回答精度維持のため）
            fallback_docs = []
            for doc in documents[:top_k]:
                d = doc.copy()
                if 'parent_content' in d.get('metadata', {}):
                    d['content'] = d['metadata']['parent_content']
                fallback_docs.append(d)
            return fallback_docs
            
        except Exception as e:
            logger.error(f"Unexpected rerank error: {e}")
            # 予期せぬエラー時も同様にフォールバック
            fallback_docs = []
            for doc in documents[:top_k]:
                d = doc.copy()
                if 'parent_content' in d.get('metadata', {}):
                    d['content'] = d['metadata']['parent_content']
                fallback_docs.append(d)
            return fallback_docs

    # ----------------------------------------------------------------
    # Step 3: LitM対策 (配置最適化)
    # ----------------------------------------------------------------
    @traceable(name="Step3_LitM_Reorder", run_type="tool")
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

    # ----------------------------------------------------------------
    # Step 4: 多様性フィルタリング
    # ----------------------------------------------------------------
    @traceable(name="Step4_Diversity_Filter", run_type="tool")
    def filter_diversity(self, documents: List[Dict], threshold: float = 0.65) -> List[Dict]:
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