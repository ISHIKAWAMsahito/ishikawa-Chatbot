import logging
import os
from supabase import create_client, Client
# core.config から必要な変数をインポート
# main.py 等との整合性を保つため、config経由で取得します
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
        
        Args:
            collection_name (str): コレクション名（現在は 'documents' テーブル全体をカウント）
            
        Returns:
            int: チャンク総数
        """
        if not self.client:
            logger.warning("⚠️ Database client is not initialized.")
            return 0
            
        try:
            # head=True, count='exact' を指定することで、
            # データの中身を取得せずに件数だけを高速に取得します。
            # collection_name に基づくフィルタリングが必要な場合は、
            # .eq('metadata->>collection', collection_name) などを追加しますが、
            # 現時点の schema.md に従い、単純なテーブルカウントを行います。
            response = self.client.table("documents").select("*", count="exact", head=True).execute()
            
            # response.count が None の場合は 0 を返す
            return response.count if response.count is not None else 0
            
        except Exception as e:
            logger.error(f"❌ Error counting chunks in collection '{collection_name}': {e}")
            return 0

# シングルトンインスタンスを作成
db_client = DatabaseClient()