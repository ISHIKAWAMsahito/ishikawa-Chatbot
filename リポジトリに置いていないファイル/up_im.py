import os
import mimetypes
import urllib.parse
from supabase import create_client
from dotenv import load_dotenv

# ----------------------------------------------------------------
# 1. 画像が入っているフォルダのリスト
# ----------------------------------------------------------------
SOURCE_DIRS = [
    r"C:\dev\ishikawa-Chatbot\チャットボットdb-20260117T104533Z-3-001\チャットボットdb\converted_images_common",
    r"C:\dev\ishikawa-Chatbot\チャットボットdb-20260117T104533Z-3-001\チャットボットdb\converted_images_rules"
]

BUCKET_NAME = "images"

# ----------------------------------------------------------------
# 2. 環境変数の読み込み
# ----------------------------------------------------------------
env_path = r"C:\dev\ishikawa-Chatbot\ishikawa-Chatbot.env"
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ エラー: 環境変数が設定されていません")
    exit()

def upload_images():
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    print(f"🚀 アップロードを開始します (ダブルエンコード適用)...")
    print(f"Target Bucket: {BUCKET_NAME}\n")
    
    total_success = 0
    total_fail = 0

    for folder_path in SOURCE_DIRS:
        if not os.path.exists(folder_path):
            print(f"⚠️ フォルダが見つかりません (スキップ): {folder_path}")
            continue

        print(f"📂 フォルダ処理中: {os.path.basename(folder_path)}")
        files = os.listdir(folder_path)
        
        for i, filename in enumerate(files):
            # 隠しファイルやディレクトリはスキップ
            if filename.startswith('.') or os.path.isdir(os.path.join(folder_path, filename)):
                continue

            file_path = os.path.join(folder_path, filename)
            
            # MIMEタイプの判定
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                mime_type = "application/octet-stream"

            # ★重要: ダブルエンコード処理
            # 1回目: 日本語 -> %E6%96... (これだとサーバー側で戻されてエラーになる)
            single_encoded = urllib.parse.quote(filename)
            # 2回目: % -> %25 (これでサーバー側がデコードしても %E6%96... という安全な文字列になる)
            double_encoded = urllib.parse.quote(single_encoded)

            print(f"   Uploading: {filename[:10]}... -> {double_encoded[:20]}... ", end="")

            try:
                with open(file_path, 'rb') as f:
                    # uploadメソッドに渡す
                    supabase.storage.from_(BUCKET_NAME).upload(
                        path=double_encoded, 
                        file=f,
                        file_options={"content-type": mime_type, "upsert": "true"}
                    )
                print("✅ OK")
                total_success += 1
            except Exception as e:
                error_msg = str(e)
                if "The resource already exists" in error_msg:
                     print("ℹ️ 既存 (Skip)")
                else:
                    print(f"❌ Failed: {error_msg}")
                    total_fail += 1

    print("-" * 50)
    print(f"🎉 完了しました！")
    print(f"成功: {total_success} 件 / 失敗: {total_fail} 件")

if __name__ == "__main__":
    upload_images()