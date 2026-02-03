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
    # Step 1: クエリ拡張 (高精度化)
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
        """
        ユーザーの曖昧な質問を、検索エンジンに最適化された明確なクエリに変換する。
        """
        cot_prompt = f"""
        あなたは世界最高峰の検索エンジニアです。以下のユーザーの質問に対して、最適な検索クエリを作成してください。
        【プロセス】
        1. ユーザーの質問の「真の意図」と「不足している前提知識」を深く分析してください。
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
    # Step 2: AIリランク (品質重視・深層分析モード)
    # ----------------------------------------------------------------
    @traceable(name="Step2_AI_Rerank", run_type="chain")
    async def rerank(self, query: str, documents: List[Dict], top_k: int) -> List[Dict]:
        """
        Step 2: リランク
        - 精度優先: 上位候補に対し、LLMを用いて意味論的な適合度を厳密に評価。
        - 文脈保持: 評価には要約を用いつつ、最終回答には「親コンテンツ（全文）」を使用し情報の欠落を防ぐ。
        - 堅牢性: 一時的な通信負荷等で時間がかかっても諦めず、確実な品質担保のためにリトライを行う。
        """
        if not documents:
            return []

        # 1. 候補の厳選 (Pre-Selection)
        # 検索スコア上位のドキュメントにリソースを集中させ、深層分析を行う。
        # 全量を処理するのではなく、有望なトップ10件を徹底的に精査する方針とする。
        initial_candidates = documents[:10]

        # 2. 分析用スニペットの生成
        candidates_text = ""
        for i, doc in enumerate(initial_candidates):
            meta = doc.get('metadata', {})
            content_for_llm = meta.get('parent_content', doc.get('content', ''))
            
            # 情報密度が高い先頭部分(300文字)を抽出。
            # ※重要なキーワードの含有率を高め、LLMが論理構成を効率的に把握できるようにする。
            snippet = content_for_llm[:300].replace('\n', ' ') 
            candidates_text += f"Document ID: {i}\nSource: {meta.get('source', 'Unknown')}\nContent: {snippet}...\n\n"

        # 3. 高精度評価プロンプト
        rerank_prompt = f"""
        あなたは大学の学務・事務に精通した高度な検索専門家です。
        以下の質問に対し、提供された「文書の抜粋」が回答の根拠としてどれだけ信頼に足るかを評価してください。

        【質問】: {query}

        【評価基準 - 品質第一】
        - 正確性: 質問の核心に対する直接的な答えが含まれているか。
        - 信頼性: 文書が公式なルールや具体的な手順に言及しているか。
        - 一般的な記述よりも、具体的かつ決定的な情報を含む文書を高く評価すること。
        - 各ドキュメントに対して、0.0〜10.0のスコアで厳密に順位付けを行ってください。

        【候補ドキュメント】
        {candidates_text}

        結果は以下のJSON形式で出力し、'reason'にはその文書を選定した論理的な理由を記述してください。
        {{
            "ranked_items": [
                {{ "id": 0, "score": 9.8, "reason": "申請期限と提出先が明確に記述されており、回答に不可欠な情報源であるため" }},
                ...
            ]
        }}
        """

        try:
            # 4. 確実な実行のためのリトライ戦略
            # 応答速度よりも「処理の完遂」と「結果の品質」を最優先する。
            # 外部APIの負荷が高い場合でも、十分なバックオフ時間を設け、確実に高品質なスコアリング結果を取得するまで粘り強く試行する。
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(Exception),
                wait=wait_exponential(multiplier=2, min=5, max=60), # 最小5秒、最大60秒まで待機し、最適なタイミングでの処理成功を狙う
                stop=stop_after_attempt(5),
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
                        
                        if idx is not None and isinstance(idx, int) and 0 <= idx < len(initial_candidates):
                            # 品質の低いドキュメントは足切りし、確信度の高い情報のみを通過させる
                            if score >= PARAMS.get("RERANK_SCORE_THRESHOLD", 3.0):
                                doc = initial_candidates[idx].copy()
                                
                                # ★重要: 最終回答生成に向けて、要約ではなく「親コンテンツ(全文)」に復元する。
                                # これにより、後続のLLMがすべての文脈を把握し、リッチな回答を生成できる。
                                if 'parent_content' in doc.get('metadata', {}):
                                    doc['content'] = doc['metadata']['parent_content']

                                doc['rerank_score'] = score
                                doc['rerank_reason'] = reason
                                reranked_docs.append(doc)

                    # スコア順に並び替え、最も品質の高いドキュメントを先頭にする
                    reranked_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
                    
                    logger.info(f"Rerank completed. Quality selection: {len(reranked_docs)} docs identified.")
                    return reranked_docs[:top_k]

        except RetryError:
            logger.error("Rerank failed after extended retries. Maintaining quality by falling back to vector search order.")
            # リトライ上限に達した場合でも、システムを停止させず、可能な限り高品質なデータを返すようフォールバック処理を行う
            return self._fallback_docs(documents, top_k)
            
        except Exception as e:
            logger.error(f"Unexpected rerank error: {e}. Executing fallback strategy.")
            # 予期せぬエラー時も同様にフォールバック
            return self._fallback_docs(documents, top_k)

    def _fallback_docs(self, documents: List[Dict], top_k: int) -> List[Dict]:
        """
        フォールバック処理用ヘルパーメソッド
        エラー時であっても、親コンテンツ（全文）への差し替えを行い、情報の欠損を防ぐ。
        """
        fallback_docs = []
        # 元の検索順の上位候補を使用
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
        """
        Lost in the Middle 現象への対策。
        重要なドキュメントを先頭と末尾に配置し、LLMの認識精度を最大化する。
        """
        if not documents: return []
        dq = deque(documents)
        reordered = []
        if dq: reordered.append(dq.popleft()) # 1位を先頭へ
        temp_tail = []
        while dq:
            temp_tail.append(dq.popleft())    # 真ん中へ
            if dq:
                reordered.append(dq.popleft()) # 次点をリストへ
        # 真ん中に置いたものを末尾に反転して結合（U字型配置）
        return reordered + temp_tail[::-1]

    # ----------------------------------------------------------------
    # Step 4: 多様性フィルタリング
    # ----------------------------------------------------------------
    @traceable(name="Step4_Diversity_Filter", run_type="tool")
    def filter_diversity(self, documents: List[Dict], threshold: float = 0.65) -> List[Dict]:
        """
        類似度の高すぎる重複情報を排除し、回答の網羅性と多様性を担保する。
        """
        unique_docs = []
        for doc in documents:
            content = doc.get('content', '')
            is_duplicate = False
            for selected in unique_docs:
                # 文字列類似度判定
                sim = SequenceMatcher(None, content, selected.get('content', '')).ratio()
                if sim > threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_docs.append(doc)
        return unique_docs