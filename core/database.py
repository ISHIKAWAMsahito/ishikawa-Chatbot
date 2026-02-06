import os
import logging
from typing import List, Dict, Any, Optional

# Supabase Client
from supabase import create_client, Client

# SQLAlchemy Imports
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# 環境変数の読み込み
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")

# -----------------------------------------------------------------------------
# 1. SQLAlchemy Setup (RDBMS: 履歴・フィードバック保存用)
# -----------------------------------------------------------------------------
# Render等のPostgres URL修正 (postgres:// -> postgresql://)
# SQLAlchemy 1.4+ では postgresql:// である必要があるため
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

SessionLocal = None
Base = declarative_base()

if DATABASE_URL:
    try:
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    except Exception as e:
        logging.error(f"SQLAlchemy Engine creation failed: {e}")

# FastAPI Dependency: DBセッション取得用
def get_db():
    """
    FastAPIのDependsで使用するDBセッション生成ジェネレータ
    """
    if SessionLocal is None:
        yield None
        return

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -----------------------------------------------------------------------------
# 2. Supabase Setup (Vector Store: 文書検索用)
# -----------------------------------------------------------------------------
class DatabaseClient:
    """
    Supabaseクライアントのラッパー。
    ハイブリッド検索やFAQ検索などのRPC呼び出しを管理します。
    """
    def __init__(self, url: str, key: str):
        self.client: Optional[Client] = None
        if url and key:
            try:
                self.client = create_client(url, key)
            except Exception as e:
                logging.error(f"Supabase client connection failed: {e}")

    def search_documents_hybrid(self, collection_name: str, query_text: str, query_embedding: List[float], match_count: int = 20) -> List[Dict[str, Any]]:
        """
        SupabaseのRPC 'hybrid_search' を呼び出す
        """
        if not self.client:
            return []
        
        try:
            params = {
                "query_text": query_text,
                "query_embedding": query_embedding,
                "match_count": match_count,
                "filter": {} # 必要に応じてフィルタを追加
            }
            # RPC呼び出し
            response = self.client.rpc("hybrid_search", params).execute()
            return response.data if response.data else []
        except Exception as e:
            logging.error(f"Hybrid search failed: {e}")
            return []

    def search_fallback_qa(self, query_embedding: List[float], match_count: int = 1) -> List[Dict[str, Any]]:
        """
        FAQ検索用RPC (例: 'match_documents') を呼び出す
        """
        if not self.client:
            return []

        try:
            params = {
                "query_embedding": query_embedding,
                "match_threshold": 0.7, # 閾値は適宜調整
                "match_count": match_count
            }
            # データベース側の関数名に合わせて変更してください
            response = self.client.rpc("match_documents", params).execute()
            return response.data if response.data else []
        except Exception as e:
            logging.error(f"FAQ search failed: {e}")
            return []

# グローバルなクライアントインスタンス
db_client = DatabaseClient(SUPABASE_URL, SUPABASE_KEY)