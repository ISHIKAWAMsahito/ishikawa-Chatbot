FROM python:3.11-slim

# ステップ2: コンテナ内の作業ディレクトリを設定
WORKDIR /app

# ステップ3: 必要なファイルをコンテナにコピー
# ライブラリのリスト、アプリケーション本体、HTMLファイルをコピーします
COPY requirements.txt .
COPY main.py .
COPY admin.html .
COPY client.html .
COPY log.html .

# ステップ4: requirements.txtに記載されたライブラリをインストール
RUN pip install --no-cache-dir -r requirements.txt

# ステップ5: アプリケーションが使用するポート8000を開放
EXPOSE 8000

# ステップ6: コンテナ起動時にUvicornサーバーを実行
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]