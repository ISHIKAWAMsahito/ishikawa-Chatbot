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

class RankedItem(typing.TypedDict):
    id: int
    score: float
    reason: str

class RerankResponse(typing.TypedDict):
    ranked_items: list[RankedItem]

class SearchService:
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    async def expand_query(self, query: str) -> str:
        """Step 1: クエリ拡張"""
        try:
            full_prompt = f"{prompts.QUERY_EXPANSION}\n\n質問: {query}"
            # 簡易的にストリームメソッドを使用し結合
            stream = await self.llm.generate_stream(full_prompt)
            text = ""
            async for chunk in stream:
                if chunk.text: text += chunk.text
            return text.strip()
        except Exception as e:
            logging.error(f"Query expansion failed: {e}")
            return query

    async def rerank(self, query: str, documents: List[Dict], top_k: int) -> List[Dict]:
        """Step 2: リランク (0-10点採点)"""
        if not documents: return []

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
                if idx is not None and 0 <= idx < len(documents):
                    if score >= PARAMS["RERANK_SCORE_THRESHOLD"]:
                        doc = documents[idx]
                        doc['rerank_score'] = score
                        reranked.append(doc)
            
            reranked.sort(key=lambda x: x['rerank_score'], reverse=True)
            return reranked[:top_k]
        except Exception as e:
            logging.error(f"Rerank failed: {e}")
            return documents[:top_k]

    def reorder_litm(self, documents: List[Dict]) -> List[Dict]:
        """Step 3: Lost in the Middle対策 (凸型配置)"""
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

    def filter_diversity(self, documents: List[Dict], threshold: float = 0.7) -> List[Dict]:
        """重複排除"""
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