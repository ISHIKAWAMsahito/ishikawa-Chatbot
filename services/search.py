import json
import logging
import asyncio
from typing import List, Dict, Optional
from difflib import SequenceMatcher
from collections import deque

import typing_extensions as typing
# AsyncRetrying を追加インポート
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, AsyncRetrying, RetryError
from async_lru import alru_cache
from langsmith import traceable

from services.llm import LLMService
from services import prompts
from core.constants import PARAMS

logger = logging.getLogger(__name__)

# （型定義などは変更なし）
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
    # Step 1: クエリ拡張 (デコレータ版のまま、設定を強化)
    # ----------------------------------------------------------------
    @alru_cache(maxsize=100)
    @traceable(name="Step1_Query_Expansion", run_type="chain")
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(8), # 8回まで粘る
        reraise=True 
    )
    async def expand_query(self, query: str) -> str:
        # （中身は前回のまま変更なし）
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
    # Step 2: AIリランク (★ここを大幅修正★)
    # ----------------------------------------------------------------
    @traceable(name="Step2_AI_Rerank", run_type="chain")
    async def rerank(self, query: str, documents: List[Dict], top_k: int) -> List[Dict]:
        """
        Step 2: リランク
        内部で AsyncRetrying を使い、失敗時はクラッシュせず元のリストを返す安全設計に変更。
        """
        if not documents:
            return []

        # リランク用プロンプト作成（前回と同じ）
        candidates_text = ""
        for i, doc in enumerate(documents):
            meta = doc.get('metadata', {})
            snippet = doc.get('content', '')[:1200].replace('\n', ' ') 
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

        # ★ ここから変更：関数内でリトライループを回す
        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(Exception),
                wait=wait_exponential(multiplier=2, min=4, max=60), # 4秒〜60秒待機
                stop=stop_after_attempt(10), # ★10回まで挑戦（約2分粘る）
                reraise=True
            ):
                with attempt:
                    # ここでAPIコール
                    resp = await self.llm.generate_json(rerank_prompt, RerankResponse)
                    
                    # JSONパース処理
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
                                doc['rerank_score'] = score
                                doc['rerank_reason'] = reason
                                reranked_docs.append(doc)

                    reranked_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
                    
                    # 成功したらここでリターン
                    logger.info(f"Rerank success. Top score: {reranked_docs[0]['rerank_score'] if reranked_docs else 0}")
                    return reranked_docs[:top_k]

        except RetryError:
            # ★ 10回失敗しても諦めない（システムを落とさない）
            # エラーログは出すが、元のドキュメントをそのまま返して処理を続行させる
            logger.error("Rerank failed after 10 attempts (Resource Exhausted). Falling back to original order.")
            # 最低限の品質担保のため、元のリストをそのまま返す
            return documents[:top_k]
            
        except Exception as e:
            # その他の予期せぬエラー
            logger.error(f"Unexpected rerank error: {e}")
            return documents[:top_k]

    # ----------------------------------------------------------------
    # Step 3 & 4 (変更なし)
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