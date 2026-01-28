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
        """
        Supabase APIを実際に呼び出すヘルパーメソッド。
        例外処理を集約しています。
        """
        try:
            res = self.client.storage.from_(self.bucket_name).create_signed_url(path, SIGNED_URL_EXPIRY)
            # レスポンスの検証（ライブラリのバージョンによって挙動が異なる場合への保険）
            if res and isinstance(res, dict) and 'signedURL' in res:
                return res['signedURL']
            elif res and isinstance(res, str): # 古いバージョンや一部のエラー時用
                return res
            return None
        except Exception as e:
            # エラー発生時はログに残すが、処理は止めない
            # スタックトレースが必要な場合は logger.exception を使用する
            logger.debug(f"Supabase署名エラー (path={path}): {e}")
            return None

    async def build_references_async(self, response_text: str, sources_map: Dict[int, Dict[str, str]]) -> str:
        """
        非同期で参照リンクテキストを生成します。

        Args:
            response_text (str): AIが生成した回答本文
            sources_map (Dict): 参照元のメタデータ辞書

        Returns:
            str: Markdown形式の参照リンク集
        """
        unique_refs = []
        seen_sources = set()
        target_items: List[Tuple[int, str, str]] = []
        
        # 参照リストのフィルタリング
        for idx, src_info in sources_map.items():
            src_display = src_info.get('display', '資料')
            src_storage = src_info.get('storage', src_display)
            
            # 重複除外
            if src_storage in seen_sources:
                continue
            
            # 本文で引用されているか、または上位3件（重要資料）なら表示対象とする
            if f"[{idx}]" in response_text or idx <= 3:
                target_items.append((idx, src_display, src_storage))
                seen_sources.add(src_storage)
        
        if not target_items:
            return ""

        # 非同期ループの取得
        loop = asyncio.get_running_loop()
        
        # スレッドプールでAPIコールを並列実行
        tasks = [
            loop.run_in_executor(self.executor, self._generate_signed_url_sync, path)
            for _, _, path in target_items
        ]
        
        # 全タスクの完了を待機
        try:
            signed_urls = await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"非同期URL生成中に予期せぬエラーが発生: {e}")
            # エラー時は空リストで続行（または適切なエラーハンドリング）
            signed_urls = [None] * len(target_items)
        
        # 結果の整形（Markdown生成）
        for (idx, display_name, _), url in zip(target_items, signed_urls):
            if url:
                # 成功時：リンク付き + 有効期限注釈
                unique_refs.append(f"* [{idx}] [{display_name}]({url}) ⏳リンク有効期限:1時間")
            else:
                # 失敗時：テキストのみ表示
                unique_refs.append(f"* [{idx}] {display_name}")

        if unique_refs:
            return "\n\n## 参照元 (クリックで資料を表示)\n" + "\n".join(unique_refs)
        
        return ""