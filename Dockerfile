# 1. 軽量なベースイメージの使用
FROM python:3.11-slim

# 環境変数の設定
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 作業ディレクトリを作成
WORKDIR /app

# 修正ポイント①: OSパッケージのアップデートと必要なツールのインストール
# apt-get upgrade -y を追加し、OSレベルの脆弱性パッチを適用します
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# 修正ポイント②: 非ルートユーザーの作成
# OSレベルの脆弱性を突かれた際のリスクを低減します
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 2. 依存関係のインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. アプリケーションコードの一括コピー
COPY . .

# ディレクトリの所有者を非ルートユーザーに変更し、ユーザーを切り替え
RUN chown -R appuser:appuser /app
USER appuser

# 4. RenderのPORT環境変数に対応した実行コマンド
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]