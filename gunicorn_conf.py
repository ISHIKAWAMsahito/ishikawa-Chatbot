import multiprocessing

# サーバーがリッスンするIPアドレスとポート
bind = "0.0.0.0:8000"

# ワーカープロセスの数
workers = (multiprocessing.cpu_count() * 2) + 1

# ワーカープロセスのクラス
worker_class = "uvicorn.workers.UvicornWorker"

# --- ログ設定 (ここからが重要) ---

# ログレベル
loglevel = "info"

# アクセスログの出力先 ('-' は標準出力を意味する)
accesslog = "-"

# エラーログの出力先
errorlog = "-"

# アクセスログのフォーマット
# %t: 日時, %r: リクエストライン, %s: ステータスコード, %b: レスポンスサイズ
# %f: リファラ, %a: User-Agent
access_log_format = '%(t)s %(r)s %(s)s %(b)s "%(f)s" "%(a)s"'
