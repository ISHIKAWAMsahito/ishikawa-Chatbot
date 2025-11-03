# 1. ベースとなるPythonの公式イメージを選択
FROM python:3.11-slim

# 2. 環境変数を設定 (ログがバッファリングされないようにする)
ENV PYTHONUNBUFFERED 1

# 3. アプリケーションの作業ディレクトリを作成
WORKDIR /app

# 4. (軽量化された) requirements.txt をコピー
#    ★ このステップが速くなり、メモリ消費も減ります ★
COPY requirements.txt .

# 5. (軽量化された) ライブラリをインストール
RUN pip install --no-cache-dir -r requirements.txt

# 6. アプリケーションのコード全体 (軽量化された main.py など) をコピー
COPY . .

# 7. Renderが $PORT を正しく解釈できる「shell形式」でサーバーを起動
#    ★ これがポートとタイムアウトの問題を解決します ★
CMD gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind "0.0.0.0:$PORT"