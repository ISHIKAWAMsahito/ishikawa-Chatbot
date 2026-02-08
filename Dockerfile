# 1. 軽量なベースイメージの使用
FROM python:3.11-slim

# 環境変数の設定
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 作業ディレクトリを作成
WORKDIR /app

# 2. 依存関係のインストール（ここを最優先でキャッシュさせる）
# requirements.txtに変更がない限り、このレイヤーは再利用されます
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. アプリケーションコードの一括コピー
# .dockerignore により不要なファイル（.venv等）は除外されます。
# 個別にCOPYするよりも、1つのレイヤーにまとめる方がイメージの解凍・転送効率が上がります。
COPY . .

# 4. RenderのPORT環境変数に対応した実行コマンド
# CMDの形式を整理し、シグナルを正しく受け取れるようにします
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]