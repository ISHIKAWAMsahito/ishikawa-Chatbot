import os
import google.generativeai as genai
from dotenv import load_dotenv

# 1. 環境変数の読み込みパス
env_path = r"C:\dev\ishikawa-Chatbot\ishikawa-Chatbot.env"

def check_gemini_tier():
    print(f"--- API Tier Check Start ---")
    
    # .envの読み込み
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"✅ 設定ファイルを読み込みました: {env_path}")
    else:
        print(f"❌ .envファイルが見つかりません: {env_path}")
        return

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY が .env 内に見つかりません。")
        return

    print(f"🔍 使用中のAPIキー（末尾）: ...{api_key[-5:]}")

    # Gemini APIの設定
    genai.configure(api_key=api_key)
    
    # 判定用のテスト
    # 無料枠か有料枠かは、エラーメッセージ内の 'quota_metric' や 'quota_id' に 
    # "free_tier" という文字列が含まれているかどうかで判別できます。
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    
    try:
        print("📡 テストリクエストを送信中...")
        # 非常に短いプロンプトでテスト
        response = model.generate_content("Ping")
        
        print("✅ 疎通に成功しました。")
        print(f"レスポンス: {response.text.strip()}")
        print("\n【診断結果】")
        print("現在は制限に達していないため正常に動いていますが、")
        print("先ほど教授に発生した429エラーに 'free_tier_requests' とあったため、")
        print("このキーは実質的に「無料枠」として扱われています。")

    except Exception as e:
        error_msg = str(e)
        print(f"\n⚠️ エラーを検知しました:")
        print("-" * 50)
        print(error_msg)
        print("-" * 50)
        
        if "free_tier" in error_msg.lower():
            print("\n🚨 【判定結果】: このAPIキーは現在「無料枠 (Free Tier)」です。")
            print("Google AI Studioの設定で、Pay-as-you-goへのアップグレードが必要です。")
        elif "429" in error_msg:
            print("\n🚨 【判定結果】: クォータ制限（リクエスト過多）ですが、無料/有料の判別ができません。")
        else:
            print("\n❓ その他のエラーが発生しました。")

if __name__ == "__main__":
    check_gemini_tier()