import os
import asyncio
from supabase import create_client
from dotenv import load_dotenv

# ----------------------------------------------------------------
# 1. 環境設定
# ----------------------------------------------------------------
# config.py と同様のロジックで読み込み
env_path = r"C:\dev\ishikawa-Chatbot\ishikawa-Chatbot.env"
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
BUCKET_NAME = "images"

if not SUPABASE_URL or not SUPABASE_KEY:
    print("エラー: 環境変数が設定されていません")
    exit()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

async def check_links():
    print(f"🔍 紐付けチェックを開始します... (Target Bucket: {BUCKET_NAME})")
    
    # 2. DBからドキュメントを取得 (metadataにsourceが含まれるもの)
    #    データ量が多い場合は limit を調整してください
    try:
        res = supabase.table("documents").select("id, metadata").limit(50).execute()
        documents = res.data
    except Exception as e:
        print(f"❌ DB接続エラー: {e}")
        return

    print(f"📄 チェック対象ドキュメント数: {len(documents)} 件\n")

    success_count = 0
    fail_count = 0

    # 3. 各ドキュメントについてStorageを確認
    for doc in documents:
        doc_id = doc.get('id')
        meta = doc.get('metadata', {})
        source_name = meta.get('source')

        if not source_name:
            print(f"⚠️  [ID:{doc_id}] sourceメタデータがありません -> スキップ")
            continue

        # チャットロジックと同じ検索ルールを再現
        candidates = [source_name]
        
        # .txt の場合は 画像拡張子も候補に入れる (chat_logic.py のロジック)
        if source_name.endswith(".txt"):
            base = os.path.splitext(source_name)[0]
            candidates.extend([f"{base}.png", f"{base}.jpg", f"{base}.jpeg", f"{base}.pdf"])

        found_file = None
        
        # 候補となるファイル名で署名付きURLが作れるか（＝存在するか）チェック
        for filename in candidates:
            try:
                # 存在確認のため短時間のURLを発行してみる
                check = supabase.storage.from_(BUCKET_NAME).create_signed_url(filename, 10)
                # エラーがなくURLが返ってくれば存在するとみなす
                if check and isinstance(check, dict) and 'signedURL' in check:
                    found_file = filename
                    break
                elif isinstance(check, str): # バージョン差異対応
                    found_file = filename
                    break
            except:
                continue

        # 結果表示
        if found_file:
            print(f"✅ [ID:{doc_id}] リンクOK: '{source_name}' -> Storage: '{found_file}'")
            success_count += 1
        else:
            print(f"❌ [ID:{doc_id}] リンク切れ: '{source_name}' (候補: {candidates} がStorageに見つかりません)")
            fail_count += 1

    print("-" * 40)
    print(f"結果: OK {success_count}件 / NG {fail_count}件")

if __name__ == "__main__":
    asyncio.run(check_links())