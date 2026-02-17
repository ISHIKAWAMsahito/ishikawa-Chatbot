import json
import logging
from typing import List, Dict, Any, Optional

# LangSmith / Utils
from langsmith import traceable
from core.constants import PARAMS
from services.storage import db_client
from models.schemas import SearchResult, DocumentMetadata
from services.llm import LLMService
from services.prompts import QUERY_EXPANSION, RERANK

# ロガー設定
logger = logging.getLogger(__name__)

class SearchService:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.supabase = db_client.client  # Supabaseクライアントへの参照

    @traceable(name="Query_Expansion", run_type="chain")
    async def expand_query(self, query: str) -> str:
        """
        ユーザーのクエリを検索用に拡張します。
        """
        # 単純すぎるクエリや短すぎる場合は拡張しないガードを入れることも可能
        if len(query) < 3:
            return query

        prompt = QUERY_EXPANSION.format(query=query)
        try:
            # ストリームではなく一括生成でキーワードを取得
            response_stream = await self.llm_service.generate_stream(prompt)
            # ストリームからテキストを結合
            expanded_text = ""
            async for chunk in response_stream:
                text = chunk.text if hasattr(chunk, 'text') else str(chunk)
                expanded_text += text
            
            cleaned_text = expanded_text.strip()
            logger.info(f"Query expansion: {query} -> {cleaned_text}")
            return cleaned_text
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return query

    @traceable(name="Rerank_Documents", run_type="chain")
    async def rerank(self, query: str, documents: List[SearchResult], top_k: int) -> List[SearchResult]:
        """
        検索結果をLLMを用いて再評価（リランク）します。
        """
        if not documents:
            return []

        # ドキュメントリストをテキスト化
        candidates_text = ""
        for i, doc in enumerate(documents):
            # 内容を要約して渡す（トークン節約）
            content_preview = doc.content[:400] + "..." if len(doc.content) > 400 else doc.content
            meta_info = f"[Source: {doc.metadata.source}]"
            candidates_text += f"ID: {i}\nContent: {content_preview}\n{meta_info}\n\n"

        prompt = RERANK.format(query=query, count=len(documents)-1, candidates_text=candidates_text)

        try:
            # JSONモードでの生成を想定 (LLMServiceの実装に依存)
            # ここでは generate_json がある前提、なければ generate_stream で代用
            # Gemini 1.5/2.0 FlashなどはJSONモード対応
            response = await self.llm_service.generate_json(prompt, None) 
            
            # レスポンスの解析 (Gemini SDKのレスポンス形式に合わせる)
            try:
                result_text = response.text
                result_json = json.loads(result_text)
                ranked_items = result_json.get("ranked_items", [])
            except Exception:
                # JSONパース失敗時は生テキストから無理やり抽出などのフォールバックが必要だが
                # ここでは簡易的に空リストまたはエラーログ
                logger.warning("Rerank JSON parse failed, returning original order.")
                return documents[:top_k]

            # スコアのマッピング
            reranked_docs = []
            for item in ranked_items:
                idx = item.get("id")
                score = item.get("score", 0.0)
                reason = item.get("reason", "")
                
                if 0 <= idx < len(documents):
                    doc = documents[idx]
                    doc.score = score
                    # 理由をメタデータに追加しておくとデバッグに便利
                    doc.metadata.rerank_reason = reason
                    reranked_docs.append(doc)

            # スコア順にソート
            reranked_docs.sort(key=lambda x: x.score, reverse=True)
            
            return reranked_docs[:top_k]

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents[:top_k]

    def filter_diversity(self, documents: List[SearchResult], similarity_threshold: float = 0.85) -> List[SearchResult]:
        """
        内容が重複しているドキュメントを除外します（MMR的な簡易処理）。
        ここでは単純に、既に選ばれたドキュメントとコンテンツが似すぎているものを弾くロジックを想定。
        ※本来はベクトル計算が必要ですが、ここでは簡易的に文字列ベースか、
          もしくはすでに計算済みの similarity を使うなどの工夫が必要です。
          今回は「IDが重複しない」かつ「内容の冒頭が完全一致しない」程度の簡易フィルタとします。
        """
        unique_docs = []
        seen_content = set()
        
        for doc in documents:
            # コンテンツの先頭50文字をハッシュ代わりに
            content_sig = doc.content[:50].strip()
            if content_sig in seen_content:
                continue
            seen_content.add(content_sig)
            unique_docs.append(doc)
            
        return unique_docs

    def reorder_litm(self, documents: List[SearchResult]) -> List[SearchResult]:
        """
        Lost In The Middle 対策:
        重要なドキュメント（スコアが高いもの）をリストの先頭と末尾に配置します。
        例: [1, 2, 3, 4, 5] -> [1, 3, 5, 4, 2] のようなイメージ
        """
        if len(documents) <= 2:
            return documents
        
        reordered = []
        # 配列をコピーして操作
        working_list = documents.copy()
        
        # 1位を先頭へ
        if working_list:
            reordered.append(working_list.pop(0))
        
        # 2位を末尾へ（最後にappendするため、今は別枠確保するか、ロジックで組む）
        # 単純な交互配置アルゴリズム
        left = []
        right = []
        
        for i, doc in enumerate(working_list):
            if i % 2 == 0:
                right.append(doc) # 偶数番目は後ろ側へ
            else:
                left.append(doc)  # 奇数番目は前側へ（ただし1位の後ろ）
                
        # 結合: [1位] + [3位, 5位...] + [Docs...] + [4位, 2位] 
        # ※本来のLITMはもっと厳密ですが、ここでは簡易実装
        # 正確には: Top, 3rd, 5th ... ... 6th, 4th, 2nd
        
        return reordered + left + right[::-1]

    @traceable(name="Search_Pipeline", run_type="chain")
    async def search(
        self, 
        query: str, 
        session_id: str, 
        collection_name: str, 
        top_k: int = 5,
        embedding_model: str = "models/gemini-embedding-001"
    ) -> Dict[str, Any]:
        """
        統合検索パイプライン
        1. クエリ拡張
        2. 資料(Documents)とFAQ(Category Fallbacks)の同時検索
        3. メタデータ補正・統合
        4. リランク
        5. 条件判定によるFAQ/資料モード分岐
        """
        
        # 1. クエリ拡張
        expanded_query = await self.expand_query(query)
        
        # 2. 埋め込みベクトル生成
        query_embedding = await self.llm_service.get_embedding(expanded_query, model=embedding_model)
        if not query_embedding:
            return {"documents": [], "is_faq_match": False}

        # --- 資料とFAQの同時検索 ---
        initial_candidates = []

        # A. 一般資料検索 (Existing RPC)
        try:
            params_doc = {
                "query_embedding": query_embedding,
                "match_threshold": 0.3, # 広めに拾う
                "match_count": 10,
                "collection_name": collection_name
            }
            # RPC呼び出し (同期処理想定)
            response_docs = self.supabase.rpc("match_unified_search", params_doc).execute()
            docs_data = response_docs.data if response_docs.data else []
            
            for item in docs_data:
                meta = item.get("metadata", {}) or {}
                # 一般資料として追加
                initial_candidates.append(SearchResult(
                    id=str(item.get('id')),
                    content=item.get('content'),
                    metadata=DocumentMetadata(
                        source=meta.get("source", "不明な資料"),
                        page=meta.get("page"),
                        chunk=meta.get("chunk"),
                        file_path=meta.get("file_path"),
                        url=meta.get("url"),
                        category=meta.get("category")
                    ),
                    similarity=item.get('similarity', 0.0)
                ))
        except Exception as e:
            logger.error(f"Document search error: {e}")

        # B. FAQ検索 (New RPC: match_category_fallbacks)
        try:
            params_faq = {
                "p_query_embedding": query_embedding,
                "p_match_threshold": 0.5, # FAQはある程度似ているものだけ
                "p_match_count": 5
            }
            response_faq = self.supabase.rpc("match_category_fallbacks", params_faq).execute()
            faq_data = response_faq.data if response_faq.data else []

            for item in faq_data:
                # FAQデータにはメタデータ構造がない場合があるため補正
                # DBから metadata JSONB が返ってくる前提だが、なければ作成
                raw_meta = item.get("metadata", {})
                if raw_meta is None: raw_meta = {}
                
                # ★重要: Python側で強制的に source="FAQ" を保証
                category_val = raw_meta.get("category") or item.get("category_name", "一般FAQ")
                
                initial_candidates.append(SearchResult(
                    id=f"faq_{item.get('id')}",
                    content=item.get('content'), # SQLで「質問:...\n回答:...」に整形済み想定
                    metadata=DocumentMetadata(
                        source="FAQ", # ここでフラグを立てる
                        category=category_val,
                        url=raw_meta.get("url")
                    ),
                    similarity=item.get('similarity', 0.0)
                ))

        except Exception as e:
            # テーブルが存在しないなどのエラー時はログを出してスルー
            logger.warning(f"FAQ search error (skippable): {e}")

        if not initial_candidates:
            return {"documents": [], "is_faq_match": False}

        # 3. リランク実行
        # FAQ候補も含めて一括で採点する (数が多い場合は top_k * 3 程度に絞ってから渡す)
        rerank_input = initial_candidates[:15]
        reranked_results = await self.rerank(query, rerank_input, top_k)

        if not reranked_results:
            return {"documents": [], "is_faq_match": False}

        # 4. 条件判定 & 分岐処理
        top_doc = reranked_results[0]
        
        # しきい値の取得
        rerank_threshold = PARAMS.get("RERANK_SCORE_THRESHOLD", 6.0)
        similarity_threshold = PARAMS.get("QA_SIMILARITY_THRESHOLD", 0.90) # 厳格な判定用

        # 判定条件
        is_faq_source = (top_doc.metadata.source == "FAQ")
        is_high_score = (top_doc.score >= rerank_threshold)
        is_high_similarity = (top_doc.similarity >= similarity_threshold)

        if is_faq_source and is_high_score and is_high_similarity:
            # --- FAQモード ---
            # 条件を満たしたFAQのみを返す（または上位のFAQのみ）
            logger.info(f"FAQ Match Triggered: Score={top_doc.score}, Sim={top_doc.similarity}")
            final_documents = [top_doc] 
            is_faq_match = True
        else:
            # --- 通常資料モード ---
            # FAQ判定に落ちた場合、通常の資料検索フローとして処理
            
            # 重複排除 (Diversity)
            filtered_docs = self.filter_diversity(reranked_results)
            
            # 再配置 (LITM)
            final_documents = self.reorder_litm(filtered_docs)
            
            is_faq_match = False
            logger.info("Standard Document Mode Triggered")

        return {
            "documents": [doc.model_dump() for doc in final_documents],
            "is_faq_match": is_faq_match
        }