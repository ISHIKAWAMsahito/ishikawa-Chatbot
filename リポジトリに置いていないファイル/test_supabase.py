import os
from supabase import create_client
from dotenv import load_dotenv

# ----------------------------------------------------------------
# 1. 環境変数の読み込み
# ----------------------------------------------------------------
env_path = r"C:\dev\ishikawa-Chatbot\ishikawa-Chatbot.env"

if os.path.exists(env_path):
    print(f"Loading env from: {env_path}")
    load_dotenv(env_path)
else:
    print(f"⚠️ Warning: File not found at {env_path}")
    load_dotenv()

# ----------------------------------------------------------------
# 2. 設定値の取得
# ----------------------------------------------------------------
URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
BUCKET = "images"  # バケット名が正しいか確認してください

# ----------------------------------------------------------------
# 3. 接続テスト実行（自動ファイル検出版）
# ----------------------------------------------------------------
def test_connection():
    print("-" * 30)
    print(f"Supabase URL: {URL}")
    print(f"Supabase Key: {'(Set)' if KEY else '(Not Set)'}")
    print("-" * 30)

    if not URL or not KEY:
        print("❌ エラー: 環境変数が正しく読み込めていません。")
        return

    try:
        # クライアント初期化
        supabase = create_client(URL, KEY)
        print("✅ クライアント初期化: OK")
        
        # A. バケット内のファイル一覧を取得してみる
        print(f"\n📂 バケット '{BUCKET}' のファイル一覧を取得中...")
        files = supabase.storage.from_(BUCKET).list()
        
        if not files:
            print(f"⚠️ バケット '{BUCKET}' は空か、存在しません。")
            print("   -> Supabase管理画面でバケットを作成し、ファイルを1つアップロードしてください。")
            return

        # 存在するファイルを1つピックアップ
        target_file = files[0]['name']
        print(f"✅ ファイルが見つかりました: {target_file}")

        # B. そのファイルで署名付きURL生成テスト
        print(f"\n🔗 '{target_file}' の署名付きURL生成を試行...")
        res = supabase.storage.from_(BUCKET).create_signed_url(target_file, 60)
        
        # 結果判定
        if isinstance(res, dict) and 'signedURL' in res:
             print(f"✅ 成功！URLが発行されました:\n{res['signedURL']}")
        elif isinstance(res, str):
             print(f"✅ 成功！URLが発行されました:\n{res}")
        else:
             print(f"❌ 失敗。レスポンス: {res}")

    except Exception as e:
        print(f"❌ エラー発生: {e}")
        if "Bucket not found" in str(e):
            print("👉 ヒント: SupabaseのStorageに 'images' という名前のバケットを作成してください。")

if __name__ == "__main__":
    test_connection()