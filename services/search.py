import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
from collections import deque

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, AsyncRetrying
from async_lru import alru_cache
from langsmith import traceable

# 必要なモジュールをインポート
from services.llm import LLMService
from services import prompts
from core.database import db_client
from core.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

# Gemini API設定
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class SearchService:
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    async def get_embedding(self, text: str, model: str = "models/gemini-embedding-001") -> List[float]:
        """テキストをベクトル化する"""
        try:
            # llm.py のメソッドを利用しても良いが、ここでは直接呼び出しも可
            # 統一感を出すなら self.llm.get_embedding を使うのがベター
            return await self.llm.get_embedding(text, model)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # 空リストを返すか再送出するかは設計次第。ここでは空リストで安全側に倒す
            return []

    # ----------------------------------------------------------------
    # Step 1: クエリ拡張
    # ----------------------------------------------------------------
    @alru_cache(maxsize=100)
    @traceable(name="Step1_Query_Expansion", run_type="chain")
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        stop=stop_after_attempt(2), # 試行回数を減らして高速化
        reraise=False # 失敗してもエラーにせず元のクエリを使う
    )
    async def expand_query(self, query: str) -> str:
        """ユーザーの質問を、検索ヒット率が高まるキーワードに変換"""
        cot_prompt = prompts.QUERY_EXPANSION.format(query=query)
        try:
            # ストリーミングではなく一括生成の方が速い場合もあるが、既存踏襲
            response = await self.llm.generate_stream(cot_prompt)
            full_text = ""
            async for chunk in response:
                if chunk.text: full_text += chunk.text
            
            expanded = full_text.strip()
            if not expanded:
                return query
            return expanded
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return query

    # ----------------------------------------------------------------
    # Step 2: AIリランク (並列処理版)
    # ----------------------------------------------------------------
    async def _check_relevance_single(self, query: str, doc: Dict, index: int) -> Optional[Dict]:
        """
        単一のドキュメントに対して関連性判定を行うヘルパー関数
        """
        meta = doc.get('metadata', {})
        content = meta.get('parent_content', doc.get('content', ''))
        # 評価用にテキストを短縮
        snippet = content[:500].replace('\n', ' ')
        
        # 1件評価用のプロンプト（軽量化）
        # ※ prompts.RERANK_SINGLE が定義されている前提、なければここで作成
        prompt = f"""
        あなたは検索エンジンのリランクシステムです。
        以下の「検索クエリ」に対して、「対象ドキュメント」がどれくらい関連しているか、0.0〜10.0のスコアで評価してください。
        
        検索クエリ: {query}
        
        対象ドキュメント:
        {snippet}
        
        出力は以下のJSON形式のみを返してください。理由(reason)は簡潔に。
        {{
            "score": 8.5,
            "reason": "具体的なキーワードが含まれているため"
        }}
        """

        try:
            # タイムアウトを短く設定したリトライ
            async for attempt in AsyncRetrying(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=3)):
                with attempt:
                    # generate_json が使えない場合（スキーマ未定義）は generate_content で JSONパース
                    model = genai.GenerativeModel("gemini-1.5-flash") # 軽量モデル指定
                    response = await model.generate_content_async(
                        prompt,
                        generation_config={"response_mime_type": "application/json"}
                    )
                    
                    text = response.text.replace("```json", "").replace("```", "").strip()
                    data = json.loads(text)
                    
                    score = float(data.get("score", 0.0))
                    
                    # 閾値チェック (6.0以上)
                    if score >= 6.0:
                        doc_copy = doc.copy()
                        if 'parent_content' in meta:
                            doc_copy['content'] = meta['parent_content']
                        doc_copy['rerank_score'] = score
                        doc_copy['rerank_reason'] = data.get("reason", "")
                        return doc_copy
                    else:
                        return None # 低スコアは捨てる
                        
        except Exception as e:
            logger.warning(f"Relevance check failed for doc {index}: {e}")
            return None # エラー時は除外（または元のドキュメントをそのまま返す）

    @traceable(name="Step2_AI_Rerank", run_type="chain")
    async def rerank(self, query: str, documents: List[Dict], top_k: int) -> List[Dict]:
        """
        AIを用いて検索結果を再ランク付けする (並列実行)
        """
        if not documents:
            return []

        # 評価対象は上位5件に限定
        initial_candidates = documents[:5]
        
        # 並列タスクの作成
        tasks = [
            self._check_relevance_single(query, doc, i)
            for i, doc in enumerate(initial_candidates)
        ]
        
        # 並列実行 (Gather)
        # return_exceptions=False なので、個別のエラーは _check_relevance_single 内で握りつぶして None を返す
        results = await asyncio.gather(*tasks)
        
        # None (低スコアまたはエラー) を除外してリスト化
        reranked_docs = [doc for doc in results if doc is not None]
        
        # スコア順にソート
        reranked_docs.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        
        # もし1件も残らなければ、フォールバックとして元の順序の上位を返す
        if not reranked_docs:
            logger.info("All documents filtered out by rerank. Fallback to vector order.")
            return self._fallback_docs(documents, top_k)
            
        return reranked_docs[:top_k]

    def _fallback_docs(self, documents: List[Dict], top_k: int) -> List[Dict]:
        """フォールバック: 親コンテンツがあれば展開して返す"""
        fallback_docs = []
        for doc in documents[:top_k]:
            d = doc.copy()
            if 'parent_content' in d.get('metadata', {}):
                d['content'] = d['metadata']['parent_content']
            fallback_docs.append(d)
        return fallback_docs

    # ----------------------------------------------------------------
    # Step 3: LitM対策
    # ----------------------------------------------------------------
    def reorder_litm(self, documents: List[Dict]) -> List[Dict]:
        if not documents: return []
        dq = deque(documents)
        reordered = []
        if dq: reordered.append(dq.popleft()) # 1位
        
        temp_tail = []
        while dq:
            temp_tail.append(dq.popleft()) # 2位 -> 末尾候補
            if dq:
                reordered.append(dq.popleft()) # 3位 -> 前半候補
        
        # 末尾候補を逆順にして結合 (U字型)
        return reordered + temp_tail[::-1]

    # ----------------------------------------------------------------
    # Step 4: 多様性フィルタリング
    # ----------------------------------------------------------------
    def filter_diversity(self, documents: List[Dict], threshold: float = 0.7) -> List[Dict]:
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

    # ----------------------------------------------------------------
    # 統合検索メソッド
    # ----------------------------------------------------------------
    @traceable(name="Search_Pipeline", run_type="chain")
    async def search(self, query: str, session_id: str, collection_name: str, top_k: int = 5) -> Dict[str, Any]:
        try:
            # 1. クエリ拡張 (失敗しても元のクエリで続行)
            expanded_query = await self.expand_query(query)
            logger.info(f"Expanded Query: {expanded_query}")

            # 2. Embedding生成
            query_embedding = await self.get_embedding(expanded_query)
            if not query_embedding:
                logger.error("Failed to generate embedding.")
                return {"documents": []}

            # 3. DB検索 (ハイブリッド)
            if not db_client.client:
                logger.error("Database client is not initialized.")
                return {"documents": []}

            match_count = 10 # リランク用に少し多めに
            params = {
                "p_query_text": expanded_query,
                "p_query_embedding": query_embedding,
                "p_match_count": match_count,
                "p_collection_name": collection_name
            }

            try:
                # ハイブリッド検索RPC呼び出し
                # ※ Supabase側で関数名が異なる場合は修正してください
                response = db_client.client.rpc("match_documents_hybrid", params).execute()
                documents = response.data
            except Exception as e:
                logger.warning(f"Hybrid search failed ({e}). Fallback to vector search.")
                # フォールバック: 純粋なベクトル検索
                vector_params = {
                    "p_query_embedding": query_embedding,
                    "p_match_count": match_count,
                    "p_collection_name": collection_name
                }
                response = db_client.client.rpc("match_documents", vector_params).execute()
                documents = response.data

            if not documents:
                return {"documents": []}

            # 4. Rerank (並列実行)
            reranked_docs = await self.rerank(query, documents, top_k)

            # 5. LitM Reorder
            reordered_docs = self.reorder_litm(reranked_docs)

            # 6. Diversity Filter
            final_docs = self.filter_diversity(reordered_docs, threshold=0.7)

            return {"documents": final_docs}

        except Exception as e:
            logger.error(f"Search pipeline error: {e}", exc_info=True)
            return {"documents": []}