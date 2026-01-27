# services/search.py
import json
import logging
import typing_extensions as typing
from collections import deque
from difflib import SequenceMatcher
from typing import List, Dict

from services.llm import LLMService
from services import prompts
from core.constants import PARAMS
from langsmith import traceable # LangSmith追跡用

class RankedItem(typing.TypedDict):
    id: int
    score: float
    reason: str

class RerankResponse(typing.TypedDict):
    ranked_items: list[RankedItem]

class SearchService:
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    @traceable(name="Step1_Query_Expansion", run_type="chain")
    async def expand_query(self, query: str) -> str:
        """Step 1: クエリ拡張 - ユーザーの質問を検索用に最適化"""
        try:
            full_prompt = f"{prompts.QUERY_EXPANSION}\n\n質問: {query}"
            stream = await self.llm.generate_stream(full_prompt)
            text = ""
            async for chunk in stream:
                if chunk.text: text += chunk.text
            return text.strip()
        except Exception as e:
            logging.error(f"Query expansion failed: {e}")
            return query

    @traceable(name="Step2_AI_Rerank", run_type="chain")
    async def rerank(self, query: str, documents: List[Dict], top_k: int) -> List[Dict]:
        """Step 2: リランク - 検索結果をAIが0-10点で採点し並べ替え"""
        if not documents: return []

        # 採点対象のリスト作成
        candidates_text = ""
        for i, doc in enumerate(documents):
            meta = doc.get('metadata', {})
            snippet = doc.get('content', '')[:300].replace('\n', ' ')
            candidates_text += f"ID:{i} [Source:{meta.get('source', '?')}]\n{snippet}\n\n"

        formatted_prompt = prompts.RERANK.format(query=query, candidates_text=candidates_text)
        
        try:
            resp = await self.llm.generate_json(formatted_prompt, RerankResponse)
            data = json.loads(resp.text)
            
            reranked = []
            for item in data.get("ranked_items", []):
                idx = item.get("id")
                score = item.get("score")
                
                # LangSmithのログに「なぜその点数にしたか」を残す工夫ができればベストですが
                # ここでは処理結果の順序を返すことでトレースに残ります
                if idx is not None and 0 <= idx < len(documents):
                    if score >= PARAMS["RERANK_SCORE_THRESHOLD"]:
                        doc = documents[idx]
                        doc['rerank_score'] = score
                        doc['rerank_reason'] = item.get("reason", "") # 理由も保持
                        reranked.append(doc)
            
            reranked.sort(key=lambda x: x['rerank_score'], reverse=True)
            return reranked[:top_k]
        except Exception as e:
            logging.error(f"Rerank failed: {e}")
            return documents[:top_k]

    @traceable(name="Step3_LitM_Reorder", run_type="tool")
    def reorder_litm(self, documents: List[Dict]) -> List[Dict]:
        """Step 3: Lost in the Middle対策 - 重要情報を両端に配置"""
        if not documents: return []
        dq = deque(documents)
        reordered = []
        if dq: reordered.append(dq.popleft()) # 1位を先頭へ
        
        temp_tail = []
        while dq:
            temp_tail.append(dq.popleft()) # 2位を後ろ候補へ
            if dq:
                reordered.append(dq.popleft()) # 3位を真ん中へ
        
        # 1位 ... 3位 ... 2位 のような凸型配置にして返す
        return reordered + temp_tail[::-1]

    @traceable(name="Step4_Diversity_Filter", run_type="tool")
    def filter_diversity(self, documents: List[Dict], threshold: float = 0.7) -> List[Dict]:
        """重複排除 - 内容が酷似しているドキュメントをカット"""
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