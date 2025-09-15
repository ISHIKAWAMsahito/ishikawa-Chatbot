# ベースイメージとしてPython 3.11のスリム版を使用
FROM python:3.11-slim-bookworm

# 作業ディレクトリを設定
WORKDIR /app

# 最初に requirements.txt をコピーして、ライブラリをインストール
# これにより、コードを変更してもライブラリの再インストールが不要になる（キャッシュが効く）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ★★★★★ ここが重要 ★★★★★
# データベースとアプリケーションのコード全体をコピー
# .dockerignore ファイルで指定されたファイル/フォルダは除外される
COPY chroma_db ./chroma_db
COPY . .

# アプリケーションがリッスンするポートを指定
# Renderは自動で80/443にマッピングするが、コンテナ内のポートを指定しておく
EXPOSE 8000

# コンテナ起動時に実行するコマンド
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
