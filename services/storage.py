# services/storage.py
import os
import asyncio
from typing import Optional, Dict, List
from supabase import create_client, Client
from concurrent.futures import ThreadPoolExecutor

from core.config import SUPABASE_URL, SUPABASE_SERVICE_KEY

class StorageService:
    def __init__(self, bucket_name: str = "images"):
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        self.bucket_name = bucket_name
        self.executor = ThreadPoolExecutor(max_workers=4)

    def _generate_signed_url_sync(self, filename: str) -> Optional[str]:
        """同期的に署名付きURLを生成"""
        try:
            res = self.client.storage.from_(self.bucket_name).create_signed_url(filename, 3600)
            if res and 'signedURL' in res:
                return res['signedURL']
        except Exception:
            pass
        
        # テキストファイルの場合、同名の画像を探すフォールバック
        if filename.endswith(".txt"):
            base_name = os.path.splitext(filename)[0]
            for ext in [".png", ".jpg", ".jpeg", ".pdf"]:
                try:
                    image_filename = f"{base_name}{ext}"
                    res = self.client.storage.from_(self.bucket_name).create_signed_url(image_filename, 3600)
                    if res and 'signedURL' in res:
                        return res['signedURL']
                except Exception:
                    continue
        return None

    async def build_references_async(self, response_text: str, sources_map: Dict[int, Dict[str, str]]) -> str:
        """非同期で参照リンクテキストを生成"""
        unique_refs = []
        seen_sources = set()
        target_items = []
        
        for idx, src_info in sources_map.items():
            src_display = src_info.get('display', '資料')
            src_storage = src_info.get('storage', src_display)
            if src_storage in seen_sources: continue
            
            # 本文で引用されているか、上位3件なら表示
            if f"[{idx}]" in response_text or idx <= 3:
                target_items.append((idx, src_display, src_storage))
                seen_sources.add(src_storage)
        
        if not target_items:
            return ""

        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(self.executor, self._generate_signed_url_sync, path)
            for _, _, path in target_items
        ]
        
        signed_urls = await asyncio.gather(*tasks)
        
        for (idx, display_name, _), url in zip(target_items, signed_urls):
            if url:
                unique_refs.append(f"* [{idx}] [{display_name}]({url}) ⏳リンク有効期限:1時間")
            else:
                unique_refs.append(f"* [{idx}] {display_name}")

        if unique_refs:
            return "\n\n## 参照元 (クリックで資料を表示)\n" + "\n".join(unique_refs)
        return ""