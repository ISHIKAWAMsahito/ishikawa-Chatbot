import os
import asyncio
from dotenv import load_dotenv
from langsmith import traceable

# 1. 指定した.envファイルを読み込む
env_path = "ishikawa-Chatbot.env"
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f"✅ {env_path} を読み込みました。")
else:
    print(f"❌ {env_path} が見つかりません。パスを確認してください。")

# 2. 設定状況の確認（セキュリティのため一部伏せ字）
def check_env(var_name):
    value = os.getenv(var_name)
    if value:
        masked = value[:6] + "..." + value[-4:] if len(value) > 10 else value
        print(f"   - {var_name}: {masked}")
    else:
        print(f"   - {var_name}: ❌ 未設定")

print("\n--- LangSmith 設定確認 ---")
check_env("LANGCHAIN_TRACING_V2")
check_env("LANGCHAIN_API_KEY")
check_env("LANGCHAIN_PROJECT")
check_env("LANGCHAIN_ENDPOINT")

# 3. 追跡テスト用関数
# 既存の llm.py や search.py の構造に似せて作成
@traceable(name="Debug_Test_Chain", run_type="chain")
async def debug_trace_test():
    print("\n--- トレース送信テスト開始 ---")
    await asyncio.sleep(1) # 通信のシミュレーション
    return "LangSmith への送信テスト成功！"

async def main():
    # 強制的にプロジェクト名を指定する場合（質問にあった内容）
    # os.environ["LANGCHAIN_PROJECT"] = "ishikawa-chatbot-eval"
    
    try:
        result = await debug_trace_test()
        print(f"結果: {result}")
        
        print("\n送信完了を確実にするため、5秒待機します...")
        await asyncio.sleep(5)
        print("完了しました。LangSmith のダッシュボードを確認してください。")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    asyncio.run(main())