#ファイルの読み書きや、リアルタイム通信（WebSocket）、ログ記録など、このプログラムに必要な道具箱をPythonやクレームワークから取り出し準備している
import os
import json
import logging
from typing import List, Optional, Dict, Any
from fastapi import WebSocket
from core.config import ACTIVE_COLLECTION_NAME, BASE_DIR

class SettingsManager:#ここから、「設定管理者という役割（クラス）を作る・このシステムの設定を一手に引き受ける責任者
    """設定管理クラス"""
    def __init__(self):
        self.settings = {
            "model": "gemini-2.5-flash",
            "collection": ACTIVE_COLLECTION_NAME,
            "embedding_model": "models/gemini-embedding-001",  # ✅ 3072次元対応モデルに変更
            "top_k": 5
        }
        self.websocket_connections: List[WebSocket] = [] #次に、現在接続しているユーザーを管理するためのリスト（名簿）を空の状態で作る
        self.settings_file = os.path.join(BASE_DIR, "shared_settings.json")#設定を保存するファイル（shared_settings.json）の場所を確認する
        self.load_settings()#最後に、以前保存した設定があればそれを読み込みに行きます（load_settingsを実行）。」

    def load_settings(self):#保存された設定ファイルがあるか確認し、あればその内容を読み取って、現在の設定を上書きする。もし読み込みに失敗しても、システムが止まらないようにエラー内容だけ記録して無視する。
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    self.settings.update(json.load(f))
        except Exception as e:
            logging.error(f"設定ファイルの読み込みエラー: {e}")

    def save_settings(self):#現在の設定内容をファイルに書き込んで保存する。これをすることで、アプリを再起動しても前の設定を覚える
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"設定ファイルの保存エラー: {e}")

    async def update_settings(self, new_settings: Dict[str, Any]):
        self.settings.update(new_settings)#メモリ上の設定を新しいものに更新する。
        self.save_settings()#上の更新をファイルに保存します。
        await self.broadcast_settings()#設定が変わりましたよ』と、接続している全員に一斉送信（ブロードキャスト）する

    async def add_websocket(self, websocket: WebSocket):#ユーザーが画面を開いて接続してきたら『接続リスト（名簿）』に追加し、ブラウザを閉じたりして切断されたらリストから削除。常に『今誰がつながっているか』を把握するための処理。
        await websocket.accept()
        self.websocket_connections.append(websocket)

    def remove_websocket(self, websocket: WebSocket):
        if websocket in self.websocket_connections:
            self.websocket_connections.remove(websocket)

    async def broadcast_settings(self):
        message = {"type": "settings_update", "data": self.settings}#新しい設定データをメッセージとして用意する
        disconnected = []#連絡がつかなくなった人リスト」を書くための、空のメモ帳を用意
        #接続リストに載っている全員に順番にそのメッセージを送ります
        for conn in self.websocket_connections:#全員分のリストを上から順にループ（繰り返し）をする
            try:#とりあえず、やってみよう」の合図。失敗するかもしれない行動をする前に置く
                await conn.send_json(message)#さっき用意した手紙（メッセージ）をその相手に送る。「届くまでちょっと待つね」という意味も含まれている
            except:#もし、さっきの送信が失敗したら（相手が電話を切っていたりしたら）」という、トラブル時の対応
                disconnected.append(conn)#送信に失敗したその相手の名前を、さっき用意した「連絡がつかなくなった人リスト」に書き込み
        for conn in disconnected:#今度は、「連絡がつかなくなった人リスト」に書かれた名前を順に見ていく
            self.remove_websocket(conn)#その人を正式な名簿から削除します。「もういないから、次からは送らないね」という処理

class ConnectionManager:#ConnectionManagerは「受付係」の役割をする部分。誰が来たら通す、帰ったら名簿から消す、全員に声をかける
    #class ConnectionManagerここからは「接続管理人」という役割の人の仕事内容（マニュアル）を書きますよ、という宣言
    def __init__(self):#この管理人がお仕事を始めるときに、最初に準備することを書く
        self.active_connections: List[WebSocket] = []#今つながっている人リストという空のノートを一冊用意します。ここに来た人の名前を書く
    
    async def connect(self, websocket: WebSocket):#誰かが接続したいと言ってきたとき」の手順
        await websocket.accept()#相手に対して「いいですよ、入ってどうぞ」と許可を出して、通話を繋ぐ
        self.active_connections.append(websocket)#その相手の名前を、「今つながっている人リスト」の最後に追加
    
    def disconnect(self, websocket: WebSocket):#誰かとの接続を切るとき（退室するとき）」の手順
        if websocket in self.active_connections:#もし、その人の名前がリストにちゃんと載っていれば」という確認
            self.active_connections.remove(websocket)#その人の名前をリストから消す
    
    async def broadcast(self, message: str):#リストにいる全員に、同じメッセージを一斉送信するとき」の手順
        for connection in self.active_connections:#リストに載っている全員に対して、一人ずつ順番に対応
            try:#送信を試してみるよ（失敗するかもだけど）」という合図
                await connection.send_text(message)#メッセージを送信
            except:#もし送信に失敗したら（相手がいなくなっていたら）」の対応
                self.disconnect(connection)#さっき作った「接続を切る手順（disconnect）」を使って、その人をリストから削除

settings_manager: Optional[SettingsManager] = None#「設定管理人」という役職の席を用意しましたが、今はまだ誰も座っていません（空席です） という状態です。後で誰かがここに割り当てられる