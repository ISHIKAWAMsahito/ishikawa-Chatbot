import asyncio
import os
import logging
from dotenv import load_dotenv

# ログを見やすく設定
logging.basicConfig(level=logging.INFO, format='%(message)s')

# --- 設定読み込み ---
env_path = r"C:\dev\ishikawa-Chatbot\ishikawa-Chatbot.env"
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f"✅ 設定ファイル: {env_path}")
else:
    load_dotenv()
    print("⚠️ 標準の .env を使用")

try:
    from core.database import SupabaseClientManager 
    from services.llm import LLMService
except ImportError as e:
    print(f"❌ モジュール読み込みエラー: {e}")
    exit(1)

TARGET_COLLECTION = "student-knowledge-base"

# ★強制カット関数（デバッグ付き）★
def force_768_debug(vector):
    original_len = len(vector)
    if original_len > 768:
        print(f"  ✂️  [補正実行] {original_len}次元 -> 768次元 にカットしました")
        return vector[:768]
    print(f"  🆗 [補正不要] 元のサイズは {original_len}次元 です")
    return vector

async def debug_run():
    print("\n🕵️‍♀️ ベクトル次元数の実態調査を開始します...")
    
    # 1. DB接続確認
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url:
        print("❌ DB設定が見つかりません")
        return
    test_db = SupabaseClientManager(url, key)
    
    # 2. LLMサービス起動
    llm = LLMService()
    test_query = "GPAの計算方法"

    print(f"\n🧪 テスト質問: '{test_query}'")
    
    try:
        # --- ステップ1: ベクトル生成 ---
        print("  ⏳ ベクトル生成中...")
        raw_emb = await llm.get_embedding(test_query)
        print(f"  📊 生成された直後の次元数: 【 {len(raw_emb)} 】")

        # --- ステップ2: 強制カット ---
        emb_final = force_768_debug(raw_emb)
        print(f"  📉 DBに送信する直前の次元数: 【 {len(emb_final)} 】")

        # --- ステップ3: DB検索 (ここでエラーが出るか確認) ---
        print("  🚀 DB検索を実行します...")
        
        docs = test_db.search_documents_by_vector(
            collection_name=TARGET_COLLECTION,
            embedding=emb_final,
            match_count=1
        )
        
        print(f"  ✅ 成功！ 検索結果: {len(docs)} 件")
        print("  🎉 結論: コードは正しく動作しています。")

    except Exception as e:
        print(f"\n❌ エラー発生！！")
        print(f"  エラー内容: {e}")
        print("  👉 もしここで '3072 and 768' と出たら、DB関数側の問題の可能性があります")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(debug_run())