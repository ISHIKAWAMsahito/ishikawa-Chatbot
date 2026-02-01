import json
import logging
import asyncio
from typing import List, Dict, Optional
from difflib import SequenceMatcher
from collections import deque

import typing_extensions as typing
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from async_lru import alru_cache  # 非同期キャッシュ用
from langsmith import traceable

# 既存のモジュールインポート
from services.llm import LLMService
from services import prompts
from core.constants import PARAMS

# ロガー設定
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# 型定義
# ----------------------------------------------------------------

class RankedItem(typing.TypedDict):
    id: int
    score: float
    reason: str

class RerankResponse(typing.TypedDict):
    ranked_items: list[RankedItem]

# ----------------------------------------------------------------
# サービス実装
# ----------------------------------------------------------------

class SearchService:
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    # ----------------------------------------------------------------
    # Step 1: クエリ拡張 (品質重視: Chain of Thought & キャッシュ)
    # ----------------------------------------------------------------
    
    @alru_cache(maxsize=100) # 同じ質問に対する拡張結果をキャッシュ
    @traceable(name="Step1_Query_Expansion", run_type="chain")
    @retry(
        retry=retry_if_exception_type(Exception),
        # 4秒, 8秒, 16秒... と待機時間を増やし、最大60秒まで待つ
        wait=wait_exponential(multiplier=2, min=4, max=60),
        # 試行回数を増やして、より粘り強くする
        stop=stop_after_attempt(8),
        reraise=True
    )
    async def expand_query(self, query: str) -> str:
        """
        Step 1: クエリ拡張
        時間をかけてでもユーザーの意図を深く読み取り、最適な検索語句を生成します。
        """
        # 品質重視のプロンプト: 思考（Reasoning）を促す
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
            # Generate streamではなく、一括生成で安定性を取る（CoTの場合は完了を待つほうが良い）
            response = await self.llm.generate_stream(cot_prompt) # generate_streamが既存メソッドのようなのでそのまま利用
            
            full_text = ""
            async for chunk in response:
                if chunk.text:
                    full_text += chunk.text
            
            expanded = full_text.strip()
            logger.info(f"Query Expanded: {query} -> {expanded}")
            return expanded

        except Exception as e:
            logger.warning(f"Query expansion failed (retrying handled by tenacity): {e}")
            raise e # retryデコレータに捕捉させるために再送出

    # ----------------------------------------------------------------
    # Step 2: AIリランク (品質重視: 詳細コンテキスト & 厳密なスコアリング)
    # ----------------------------------------------------------------

    @traceable(name="Step2_AI_Rerank", run_type="chain")
    @retry(
        wait=wait_exponential(multiplier=2, min=4, max=60), # リランクは重いので待機時間を長めに
        stop=stop_after_attempt(3)
    )
    async def rerank(self, query: str, documents: List[Dict], top_k: int) -> List[Dict]:
        """
        Step 2: リランク
        時間を優先し、各ドキュメントの「内容」をしっかり読み込んで採点します。
        """
        if not documents:
            return []

        # 品質優先: コンテキスト幅を拡大 (300文字 -> 1000文字)
        # Geminiはコンテキストウィンドウが広いため、より多くの情報を含めたほうが精度が上がる
        candidates_text = ""
        for i, doc in enumerate(documents):
            meta = doc.get('metadata', {})
            # 改行を削除してトークン密度を高める
            snippet = doc.get('content', '')[:1200].replace('\n', ' ') 
            candidates_text += f"Document ID: {i}\nSource: {meta.get('source', 'Unknown')}\nContent: {snippet}\n\n---\n\n"

        # 品質重視のプロンプト: 評価基準を明確化
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

        結果は必ず以下のJSONフォーマットで出力してください。理由(reason)は論理的に記述してください。
        {{
            "ranked_items": [
                {{ "id": 0, "score": 9.5, "reason": "..." }},
                ...
            ]
        }}
        """

        try:
            # generate_jsonの使用（LLMServiceに実装されている前提）
            resp = await self.llm.generate_json(rerank_prompt, RerankResponse)
            
            # テキストパース処理（万が一 generate_json が dict ではなくオブジェクトを返す場合に対応）
            if hasattr(resp, 'text'):
                raw_json = resp.text
            else:
                raw_json = str(resp)

            # JSONクリーニング（Markdown記法が含まれる場合の対策）
            raw_json = raw_json.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw_json)
            
            reranked_docs = []
            ranked_items = data.get("ranked_items", [])
            
            # スコア付けとフィルタリング
            for item in ranked_items:
                idx = item.get("id")
                score = float(item.get("score", 0.0))
                reason = item.get("reason", "")
                
                if idx is not None and isinstance(idx, int) and 0 <= idx < len(documents):
                    # しきい値判定 (PARAMS定数を使用)
                    if score >= PARAMS.get("RERANK_SCORE_THRESHOLD", 3.0):
                        doc = documents[idx].copy() # 元データを汚さないようコピー
                        doc['rerank_score'] = score
                        doc['rerank_reason'] = reason
                        reranked_docs.append(doc)

            # スコア降順ソート
            reranked_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            # ログ出力（デバッグ用）
            if reranked_docs:
                top_reason = reranked_docs[0].get('rerank_reason', 'N/A')
                logger.info(f"Top Doc Score: {reranked_docs[0]['rerank_score']} | Reason: {top_reason}")

            return reranked_docs[:top_k]

        except Exception as e:
            logger.error(f"Rerank failed: {e}")
            # リトライデコレータが反応するように例外を投げる
            # ただし、最後の試行で失敗した場合は、空リストではなく元のドキュメントを返すロジックが必要
            # ここでは tenacity に任せて例外を上げ、呼び出し元でキャッチするか、
            # あるいはここでフォールバックするか判断が必要。
            # 「品質優先」ならエラーを隠蔽せずリトライさせるべきなので raise します。
            raise e

    # ----------------------------------------------------------------
    # Step 3: Lost in the Middle 対策 (ロジックは変更なし、コメント補強)
    # ----------------------------------------------------------------
    @traceable(name="Step3_LitM_Reorder", run_type="tool")
    def reorder_litm(self, documents: List[Dict]) -> List[Dict]:
        """
        Step 3: Lost in the Middle対策
        LLMは「最初」と「最後」の情報を重視する傾向があるため、
        重要なドキュメント（スコア上位）を両端に配置します。
        """
        if not documents: return []
        
        # 処理効率化のためdequeを使用
        dq = deque(documents)
        reordered = []
        
        # 1位 -> 先頭
        if dq: reordered.append(dq.popleft())
        
        temp_tail = []
        while dq:
            # 2位 -> 末尾候補へ
            temp_tail.append(dq.popleft())
            if dq:
                # 3位 -> 先頭グループの次（真ん中寄り）へ
                reordered.append(dq.popleft())
        
        # [1位, 3位, 5位, ..., 6位, 4位, 2位] のような配置になる
        return reordered + temp_tail[::-1]

    # ----------------------------------------------------------------
    # Step 4: 多様性フィルタ (品質重視: 重複の厳密な排除)
    # ----------------------------------------------------------------
    @traceable(name="Step4_Diversity_Filter", run_type="tool")
    def filter_diversity(self, documents: List[Dict], threshold: float = 0.65) -> List[Dict]:
        """
        Step 4: 重複排除
        内容は似ているが少し違うドキュメントが並ぶのを防ぎます。
        thresholdを少し厳しめ(0.7 -> 0.65)にして、より多様な情報を残すように調整。
        """
        unique_docs = []
        for doc in documents:
            content = doc.get('content', '')
            is_duplicate = False
            
            for selected in unique_docs:
                selected_content = selected.get('content', '')
                
                # SequenceMatcherは重いが、Python標準で確実な文字一致率を出せる
                # 時間をかけても良いので、ここはこのまま採用
                sim = SequenceMatcher(None, content, selected_content).ratio()
                
                if sim > threshold:
                    is_duplicate = True
                    # ログに残すとチューニングしやすい
                    # logger.debug(f"Duplicate removed (Sim: {sim:.2f}): {doc.get('metadata', {}).get('source')}")
                    break
            
            if not is_duplicate:
                unique_docs.append(doc)
                
        return unique_docs