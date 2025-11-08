import logging
from typing import List, Optional, Dict, Any
from supabase import create_client, Client

class SupabaseClientManager:
    """Supabaseクライアント管理クラス"""
    def search_documents_by_vector(self, collection_name: str, embedding: List[float], match_count: int) -> List[dict]:
        """カテゴリで絞り込まずにベクトル検索を行う"""
        params = {
            "p_collection_name": collection_name,
            "p_query_embedding": embedding,
            "p_match_count": match_count
        }
        # 作成した新しいRPC関数 'match_documents_by_vector' を呼び出す
        result = self.client.rpc("match_documents_by_vector", params).execute()
        return result.data or []
    def search_fallback_qa(self, embedding: List[float], match_count: int) -> List[dict]:
        """Q&Aフォールバックをベクトル検索する"""
        params = {
            "p_query_embedding": embedding,
            "p_match_count": match_count
        }
        # ステップ1で作成した 'match_fallback_qa' を呼び出す
        result = self.client.rpc("match_fallback_qa", params).execute()
        return result.data or []
    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)

    def get_db_type(self) -> str:
        return "supabase"

    def insert_document(self, content: str, embedding: List[float], metadata: dict):
        self.client.table("documents").insert({
            "content": content,
            "embedding": embedding,
            "metadata": metadata
        }).execute()

    def search_documents(self, collection_name: str, category: str, embedding: List[float], match_count: int) -> List[dict]:
        params = {
            "p_collection_name": collection_name,
            "p_category": category,
            "p_query_embedding": embedding,
            "p_match_count": match_count
        }
        result = self.client.rpc("match_documents", params).execute()
        return result.data or []

    def get_documents_by_collection(self, collection_name: str) -> List[dict]:
        result = self.client.table("documents").select("id, metadata").eq("metadata->>collection_name", collection_name).execute()
        return result.data or []

    def count_chunks_in_collection(self, collection_name: str) -> int:
        result = self.client.table("documents").select("id", count='exact').eq("metadata->>collection_name", collection_name).execute()
        return result.count or 0

    def get_distinct_categories(self, collection_name: str) -> List[str]:
        try:
            result = self.client.rpc("get_distinct_categories", {"p_collection_name": collection_name}).execute()
            categories = [item['category'] for item in (result.data or []) if item.get('category')]
            return categories if categories else ["その他"]
        except Exception as e:
            logging.error(f"RPC 'get_distinct_categories' の呼び出しエラー: {e}")
            return ["その他"]




# --------------------------------------------------------------------------
# 4. FastAPIアプリケーションのセットアップ
# --------------------------------------------------------------------------
db_client: Optional[SupabaseClientManager] = None