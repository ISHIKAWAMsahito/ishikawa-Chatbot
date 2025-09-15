# b.py はこの内容にしてください

import google.generativeai as genai
import os

# 「GEMINI_API_KEY」という名前の環境変数を探し、その「値」を取得する
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

try:
    print("Embeddingモデルへのリクエストを1回だけ試します...")
    
    result = genai.embed_content(
        model="models/embedding-001",
        content="これはテストです。"
    )

    print("リクエスト成功！")
    print(str(result['embedding'][:5]) + "...") 

except Exception as e:
    print("エラーが発生しました。")
    print(e)