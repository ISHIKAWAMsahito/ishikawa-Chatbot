# ベースイメージとしてPython 3.11のスリム版を使用
FROM python:3.11-slim-bookworm

# 作業ディレクトリを設定
WORKDIR /app

# 最初に requirements.txt をコピーして、ライブラリをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのコード全体をコピー
COPY . .

# 必要なディレクトリを作成
RUN mkdir -p chroma_db

# アプリケーションがリッスンするポートを指定
EXPOSE 8000

# コンテナ起動時に実行するコマンド
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]