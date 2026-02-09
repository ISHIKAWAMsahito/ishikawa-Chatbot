import os
import json
import logging
from typing import List, Optional, Dict, Any
from fastapi import WebSocket
from core.config import ACTIVE_COLLECTION_NAME, BASE_DIR

class SettingsManager:
    """設定管理クラス"""
    def __init__(self):
        self.settings = {
            "model": "models/gemini-2.5-flash",
            "collection": ACTIVE_COLLECTION_NAME,
            "embedding_model": "models/gemini-embedding-001",
            "top_k": 5
        }
        # 管理者とクライアントを分けて管理
        self.admin_connections: List[WebSocket] = []
        self.client_connections: List[WebSocket] = []
        
        self.settings_file = os.path.join(BASE_DIR, "shared_settings.json")
        self.load_settings()

    def load_settings(self):
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    self.settings.update(json.load(f))
        except Exception as e:
            logging.error(f"設定ファイルの読み込みエラー: {e}")

    def save_settings(self):
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"設定ファイルの保存エラー: {e}")

    async def update_settings(self, new_settings: Dict[str, Any]):
        self.settings.update(new_settings)
        self.save_settings()
        await self.broadcast_settings()

    async def add_websocket(self, websocket: WebSocket, is_admin: bool = False):
        """
        WebSocket接続を追加し、即座に現在の設定を送信する
        """
        await websocket.accept()
        
        # 接続直後に現在の設定を送信
        try:
            message = {"type": "settings_update", "data": self.settings}
            await websocket.send_json(message)
        except Exception as e:
            logging.error(f"初期設定送信エラー: {e}")
            return

        if is_admin:
            self.admin_connections.append(websocket)
        else:
            self.client_connections.append(websocket)

    def remove_websocket(self, websocket: WebSocket):
        if websocket in self.admin_connections:
            self.admin_connections.remove(websocket)
        if websocket in self.client_connections:
            self.client_connections.remove(websocket)

    async def broadcast_settings(self):
        """全接続（管理者・クライアント）に新しい設定を配信"""
        message = {"type": "settings_update", "data": self.settings}
        
        # 管理者へ送信
        disconnected_admin = []
        for conn in self.admin_connections:
            try:
                await conn.send_json(message)
            except:
                disconnected_admin.append(conn)
        for conn in disconnected_admin:
            if conn in self.admin_connections:
                self.admin_connections.remove(conn)

        # クライアントへ送信
        disconnected_client = []
        for conn in self.client_connections:
            try:
                await conn.send_json(message)
            except:
                disconnected_client.append(conn)
        for conn in disconnected_client:
            if conn in self.client_connections:
                self.client_connections.remove(conn)