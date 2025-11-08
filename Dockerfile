# Dockerfile
FROM python:3.11-slim

# 環境変数を設定
ENV PYTHONUNBUFFERED=1

# 作業ディレクトリを作成
WORKDIR /app

# requirements.txtをコピーしてインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーション全体をコピー（分割された構造に対応）
COPY main.py .
COPY core/ ./core/
COPY models/ ./models/
COPY services/ ./services/
COPY api/ ./api/
COPY static/ ./static/

# Renderが $PORT を正しく解釈できるようにサーバーを起動
CMD uvicorn main:app --host 0.0.0.0 --port $PORT