import logging
import os
from supabase import create_client, Client
# core.config から必要な変数をインポート
from core.config import SUPABASE_URL, SUPABASE_SERVICE_KEY

logger = logging.getLogger(__name__)

class DatabaseClient:
    """
    Supabase接続を管理するシングルトンクライアント
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseClient, cls).__new__(cls)
            cls._instance.client = None
        return cls._instance

    def __init__(self):
        # 二重初期化防止
        if self.client is None:
            self.initialize()

    def initialize(self):
        """Supabaseクライアントの初期化"""
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            logger.error("❌ SUPABASE_URL or SUPABASE_SERVICE_KEY is missing.")
            return

        try:
            self.client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
            logger.info("✅ Supabase client initialized.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase client: {e}")

    def count_chunks_in_collection(self, collection_name: str) -> int:
        """
        指定されたコレクション（documentsテーブル）内のチャンク数をカウントする。
        """
        if not self.client:
            logger.warning("⚠️ Database client is not initialized.")
            return 0
            
        try:
            response = self.client.table("documents").select("*", count="exact", head=True).execute()
            return response.count if response.count is not None else 0
        except Exception as e:
            logger.error(f"❌ Error counting chunks in collection '{collection_name}': {e}")
            return 0

# シングルトンインスタンスを作成
db_client = DatabaseClient()

# ★追加: api/chat.py 等で使用される依存性注入用の関数
def get_db():
    """
    FastAPIのDependsで使用される依存関係関数。
    シングルトンのdb_clientインスタンスを返します。
    """
    return db_client