# 必要なライブラリをインポートします
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import re
import random
import json
import os  # osライブラリをインポート
from dotenv import load_dotenv # .envファイルを読み込むために追加

# .envファイルから環境変数を読み込みます
load_dotenv()

# Flaskアプリケーションを初期化
# static_folder=Noneにすることで、Flaskのデフォルトの/staticルートを無効化できます
app = Flask(__name__)
# CORSを有効にし、すべてのオリジンからのリクエストを許可
CORS(app)

# 環境変数からGemini APIキーを取得
API_KEY = os.getenv("GEMINI_API_KEY")
# 使用するモデルを指定
MODEL_NAME = "gemini-1.5-flash-latest" 
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"


# --- Webページ配信 ---
@app.route('/')
def serve_othello():
    """
    ルートURLにアクセスされた際にothello.htmlを返す
    """
    # othello.htmlがserver.pyと同じディレクトリにあることを想定
    return send_from_directory('.', 'othello.html')


# --- APIロジック ---
def get_ai_response(board_text, valid_moves_str):
    """
    Gemini APIを呼び出して、AIに次の手を決定させ、その思考を返します。
    """
    if not API_KEY:
        # APIキーが設定されていない場合はエラーメッセージを返す
        return "エラー: GEMINI_API_KEYが設定されていませんにゃん。"

    # Geminiに渡すプロンプト
    prompt = f"""
あなたは猫のオセロプレイヤー「オセロにゃん」です。以下の盤面とルールを厳密に守り、あなたの最善の手を一つだけ答えてくださいにゃん。
語尾には必ず「にゃん」や「にゃ」をつけてくださいにゃ。

# ルール
- あなたの手番は「白(W)」ですにゃん。
- 石は必ず、相手の石を1つ以上ひっくり返せる場所にのみ置くことができますにゃ。
- あなたが置ける有効な場所は、以下のリストに限られますにゃん: [{valid_moves_str}]

# 現在の盤面:
{board_text}

# 指示
あなたの思考プロセスを簡潔に述べた後、最終的な結論として `MOVE: [座標]` の形式で着手する座標を答えてくださいにゃん。
例えば、「角が取れるからF5が有利だにゃん。MOVE: F5」のように答えてくださいにゃ。
座標は必ず、上記の有効な場所リスト [{valid_moves_str}] の中から選んでくださいにゃん。
"""

    # Gemini APIに送信するデータを作成
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 256
        }
    }
    headers = {'Content-Type': 'application/json'}

    try:
        # Gemini APIにPOSTリクエストを送信
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # エラーがあれば例外を発生させる
        
        data = response.json()
        
        # 応答からテキスト部分を抽出
        content = data['candidates'][0]['content']['parts'][0]['text']
        
        # AIの応答に "MOVE:" が含まれているか、かつその手が有効かチェック
        match = re.search(r'MOVE:\s*([A-H][1-8])', content, re.IGNORECASE)
        if match:
            move_str = match.group(1).upper()
            if move_str in valid_moves_str.upper().split(', '):
                return content  # 有効ならAIの応答をそのまま返す
            else:
                print(f"AIが無効な手を選択しました: {move_str}")
        else:
            print(f"AIからの応答形式が不正です: {content}")

    except requests.exceptions.RequestException as e:
        print(f"Gemini APIとの通信中にエラーが発生しました: {e}")
    except (KeyError, IndexError) as e:
        print(f"APIからの応答形式が予期せぬものです: {e}")
        print(f"受信データ: {data}")


    # AIが適切な手を返せなかった場合、ランダムな手で応答を生成する
    print("AIが手を決定できなかったため、ランダムな手で応答を生成します。")
    valid_moves_list = valid_moves_str.split(', ')
    random_move = random.choice(valid_moves_list)
    return f"うーん、良い手が見つからにゃいから、ランダムに選ぶにゃん。MOVE: {random_move}"


# APIエンドポイントの定義
@app.route('/api/get_ai_move', methods=['POST'])
def handle_ai_move():
    """
    フロントエンドからのリクエストを受け取り、AIの応答を返すエンドポイント
    """
    try:
        data = request.json
        board_text = data.get('board_text')
        valid_moves_str = data.get('valid_moves_str')

        if not board_text or not valid_moves_str:
            return jsonify({'detail': 'board_textとvalid_moves_strが必要です。'}), 400

        # AIに応答を生成させる
        ai_response_text = get_ai_response(board_text, valid_moves_str)

        return jsonify({'response_text': ai_response_text})

    except Exception as e:
        print(f"API処理中にエラーが発生しました: {e}")
        return jsonify({'detail': 'サーバー内部でエラーが発生しました。'}), 500

# サーバーを起動
if __name__ == '__main__':
    if not API_KEY:
        print("="*50)
        print("警告: 環境変数 'GEMINI_API_KEY' が設定されていません。")
        print(".envファイルを作成し、'GEMINI_API_KEY=あなたのAPIキー' を記述してください。")
        print("="*50)
    
    app.run(port=5001, debug=True)
