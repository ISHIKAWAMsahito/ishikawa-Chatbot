# ベースを互換性の高いDebianベースのslimイメージに変更
FROM python:3.11-slim-bookworm

# コンテナ内の作業ディレクトリを設定
WORKDIR /app

# 必要なライブラリをインストールするためのファイルをコピー
COPY requirements.txt .

# 脆弱性対策とビルド依存関係のインストール
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends build-essential gcc && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y --auto-remove build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# アプリケーションのソースコードをすべてコピー  <--- ★まず、すべてのファイルをコピー
COPY . .

# ★次に、コピーされたファイルの一覧を出力
RUN ls -la

# コンテナの10000番ポートを外部に公開することを宣言
EXPOSE 10000

# コンテナが起動したときに実行されるコマンド
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]