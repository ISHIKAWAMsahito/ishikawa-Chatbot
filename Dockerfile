# 1. 最新のパッチが適用されたスリムイメージを使用
FROM python:3.11-slim

# 環境変数の設定
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 修正ポイント①: OSパッケージのアップデートを強制する
# apt-get upgrade を入れることで、Debian公式が配布している最新のセキュリティパッチ(glibcやopenssl等)を適用します
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# 修正ポイント②: 非ルートユーザーの作成
# root権限での実行を避けることで、OSレベルの脆弱性（systemd等）を突かれた際の被害を抑えます
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 2. 依存関係のインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt [cite: 2]

# 3. アプリケーションコードのコピー
COPY . .

# 所有者を非ルートユーザーに変更
RUN chown -R appuser:appuser /app
USER appuser

# 4. 実行コマンド
# ポートはRender等の環境変数に従う [cite: 3]
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]