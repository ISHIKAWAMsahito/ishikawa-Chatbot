# services/storage.py
import os
import asyncio
import logging
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor

from supabase import create_client, Client
from core.config import SUPABASE_URL, SUPABASE_SERVICE_KEY

# ロガーの設定（モジュール単位で設定）
logger = logging.getLogger(__name__)

# 定数設定（将来的には core.config から読み込むことを推奨）
DEFAULT_BUCKET_NAME = "images"
SIGNED_URL_EXPIRY = 3600  # 1時間 (秒)

class StorageService:
    """
    Supabase Storage を操作し、セキュアな署名付きURLを発行するサービスクラス。
    非同期処理とスレッドプールを用いて、I/O待ちによるパフォーマンス低下を防ぎます。
    """

    def __init__(self, bucket_name: str = DEFAULT_BUCKET_NAME):
        """
        Args:
            bucket_name (str): 対象のStorageバケット名
        """
        # SERVICE_KEYの使用は管理者権限となるため、取り扱いに注意が必要です
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        self.bucket_name = bucket_name
        # 外部APIコール用のスレッドプール（同時実行数を制御）
        self.executor = ThreadPoolExecutor(max_workers=4)

    def _generate_signed_url_sync(self, filename: str) -> Optional[str]:
        """
        署名付きURLを同期的に生成します。
        指定されたファイルが見つからない場合、拡張子を変更して再試行（フォールバック）します。

        Args:
            filename (str): ストレージ上のファイルパス

        Returns:
            Optional[str]: 署名付きURL（生成失敗時は None）
        """
        # 開発・デバッグ時のみ詳細を出力（本番ログレベルでは表示されない想定）
        logger.debug(f"URL生成試行開始: {filename}")
        
        # 1. 指定されたファイル名で生成を試行
        url = self._call_supabase_sign(filename)
        if url:
            return url

        # 2. テキストファイルの場合のフォールバック処理（画像探索）
        if filename.endswith(".txt"):
            base_name = os.path.splitext(filename)[0]
            # 探索する拡張子のリスト
            fallback_extensions = [".png", ".jpg", ".jpeg", ".pdf"]
            
            for ext in fallback_extensions:
                image_filename = f"{base_name}{ext}"
                logger.debug(f"画像フォールバック試行: {image_filename}")
                
                url = self._call_supabase_sign(image_filename)
                if url:
                    logger.info(f"画像フォールバック成功: {filename} -> {image_filename}")
                    return url
        
        logger.warning(f"URL生成失敗（対象ファイルなし）: {filename}")
        return None

    def _call_supabase_sign(self, path: str) -> Optional[str]:
        """Supabase APIを呼び出し署名付きURLを発行する（単一パス用）"""
        try:
            res = self.client.storage.from_(self.bucket_name).create_signed_url(path, SIGNED_URL_EXPIRY)
            if res and isinstance(res, dict) and 'signedURL' in res:
                return res['signedURL']
            elif res and isinstance(res, str):
                return res
            return None
        except Exception as e:
            logger.debug(f"Supabase署名エラー (path={path}): {e}")
            return None

    def _get_multiple_urls_sync(self, source_name: str) -> List[str]:
        """
        DBから対象の全画像パスを取得し、それぞれの署名付きURLを生成します。
        """
        urls = []
        try:
            # 1. DBからメタデータを取得
            # TABLE_NAME は "documents" を想定
            res = self.client.table("documents").select("metadata") \
                .eq("metadata->>source", source_name).limit(1).execute()
            
            if not res.data:
                # DBにない場合は、ファイル名そのもので1回試行（フォールバック）
                url = self._call_supabase_sign(source_name)
                return [url] if url else []

            meta = res.data[0].get('metadata', {})
            if isinstance(meta, str):
                import json
                meta = json.loads(meta)

            # 2. パスリストの作成
            # image_paths (リスト) を優先し、なければ image_path (文字列) を使う
            paths = meta.get('image_paths', [])
            if not paths and meta.get('image_path'):
                paths = [meta['image_path']]
            
            # 3. 各パスに対してURLを発行
            for p in paths:
                u = self._call_supabase_sign(p)
                if u:
                    urls.append(u)
                    
            return urls
        except Exception as e:
            logger.error(f"URL生成同期処理エラー: {e}")
            return []

    async def build_references_async(self, response_text: str, sources_map: Dict[int, Dict[str, str]]) -> str:
        """
        非同期で参照リンクテキストを生成します（複数画像対応版）。
        """
        unique_refs = []
        seen_sources = set()
        target_items: List[Tuple[int, str, str]] = []
        
        for idx, src_info in sources_map.items():
            src_display = src_info.get('display', '資料')
            src_storage = src_info.get('storage', src_display)
            
            if src_storage in seen_sources:
                continue
            
            if f"[{idx}]" in response_text or idx <= 3:
                target_items.append((idx, src_display, src_storage))
                seen_sources.add(src_storage)
        
        if not target_items:
            return ""

        loop = asyncio.get_running_loop()
        
        # 各アイテムに対して「URLのリスト」を取得するタスクを生成
        tasks = [
            loop.run_in_executor(self.executor, self._get_multiple_urls_sync, path)
            for _, _, path in target_items
        ]
        
        try:
            # List[List[str]] (アイテムごとのURLリストのリスト) が返ってくる
            urls_groups = await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"非同期URL生成中に予期せぬエラーが発生: {e}")
            urls_groups = [[]] * len(target_items)
        
        # 結果の整形
        for (idx, display_name, _), urls in zip(target_items, urls_groups):
            if urls:
                # リンクの生成: 1枚なら「資料名」、複数なら「資料名 (表示1, 表示2...)」
                if len(urls) == 1:
                    unique_refs.append(f"* [{idx}] [{display_name}]({urls[0]}) ⏳有効期限:1時間")
                else:
                    links = " ".join([f"[表示{i+1}]({u})" for i, u in enumerate(urls)])
                    unique_refs.append(f"* [{idx}] {display_name} ({links}) ⏳有効期限:1時間")
            else:
                unique_refs.append(f"* [{idx}] {display_name}")

        if unique_refs:
            return "\n\n## 参照元 (クリックで資料を表示)\n" + "\n".join(unique_refs)
        
        return ""