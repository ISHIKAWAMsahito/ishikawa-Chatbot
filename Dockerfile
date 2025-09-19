# ステップ1: ベースイメージを選択
# Python 3.11 の軽量版を土台として使用します
FROM python:3.11-slim

# ステップ2: 作業ディレクトリを設定
# コンテナ内の /app ディレクトリを作成し、以降のコマンドはここを基準に実行します
WORKDIR /app

# ステップ3: 必要なファイルをコンテナにコピー
# まず、ライブラリのリスト(requirements.txt)をコピーします
COPY requirements.txt .

<<<<<<< HEAD

=======
# ▼▼▼【ここが最重要ポイントです】▼▼▼
# 'custom_components' フォルダとその中身を、丸ごとコンテナにコピーします
# これにより、main.py が 'from custom_components...' というインポート文を解決できるようになります
COPY custom_components/ ./custom_components/
>>>>>>> 60d72e17b4e23974c67d915281db7df0c640ad59

# アプリケーション本体とHTMLファイルをコピーします
COPY main.py .
COPY admin.html .
# ▲▲▲【コピーコマンドここまで】▲▲▲

# ステップ4: ライブラリをインストール
# requirements.txt に書かれたライブラリをコンテナ内にインストールします
RUN pip install --no-cache-dir -r requirements.txt

# ステップ5: ポートを開放
# FastAPIサーバーが使用するコンテナの8000番ポートを外部に公開します
EXPOSE 8000

# ステップ6: アプリケーションを実行
# コンテナが起動したときに、このコマンドが自動的に実行されます
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
