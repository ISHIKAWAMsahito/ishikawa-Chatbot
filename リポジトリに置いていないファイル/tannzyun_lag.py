import pandas as pd
import asyncio
import os
from dotenv import load_dotenv

# 1. 環境設定
# 既存の .env ファイルを読み込みます
load_dotenv(r"C:\dev\ishikawa-Chatbot\ishikawa-Chatbot.env")

# 必要なモジュールをインポート
try:
    from core.database import SupabaseClientManager 
    from services.llm import LLMService
    from services.search import SearchService
    from services import prompts 
except ImportError as e:
    print(f"❌ インポートエラー: {e}")
    print("このスクリプトは 'evaluate.py' と同じフォルダに置いて実行してください。")
    exit(1)

# 設定
INPUT_FILE = "質問とチャットボットシステムの回答集.xlsx"
OUTPUT_FILE = "comparison_result.xlsx"

# ==========================================
# 改善前(Before)のロジック: 単純なハイブリッド検索 + 上位5件
# ==========================================
async def generate_before_answer(question: str):
    llm = LLMService()
    db = SupabaseClientManager(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
    
    try:
        # 1. 検索用埋め込みベクトル作成
        emb = await llm.get_embedding(question)
        
        # 2. 単純検索 (クエリ拡張なし、Rerankなし)
        raw_docs = db.client.rpc("match_documents_hybrid", {
            "p_collection_name": "student-knowledge-base",
            "p_query_text": question,
            "p_query_embedding": emb,
            "p_match_count": 5  # 上位5件のみ
        }).execute().data or []

        # 3. コンテキスト作成
        context_str = "\n\n".join([d.get('content','') for d in raw_docs])
        
        # 4. 回答生成
        if not context_str:
            return "検索結果なし"

        prompt = f"質問: {question}\n\n<context>\n{context_str}\n</context>"
        
        # LLMで回答生成
        res_stream = await llm.generate_stream(prompt, prompts.SYSTEM_GENERATION)
        answer = ""
        async for chunk in res_stream:
            if chunk.text: answer += chunk.text
            
        return answer.strip()

    except Exception as e:
        return f"エラー: {e}"

# ==========================================
# メイン処理
# ==========================================
async def main():
    print(f"📂 ファイル読み込み中: {INPUT_FILE}")
    try:
        df = pd.read_excel(INPUT_FILE)
        # 列名の空白削除
        df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        print(f"❌ 読み込み失敗: {e}")
        return

    results = []
    
    print("🚀 比較データの生成を開始します...")
    print(f"{'-'*60}")
    print(f"{'質問':<30} | {'進捗'}")
    print(f"{'-'*60}")

    for index, row in df.iterrows():
        question = str(row['Question']).strip()
        after_answer = str(row['Answer']).strip() # Excelにある改善後の回答

        print(f"[{index+1}/{len(df)}] {question[:20]}... ", end="", flush=True)

        # Before(改善前)の回答を生成
        before_answer = await generate_before_answer(question)
        
        print("✅ 完了")

        # 結果をリストに追加
        results.append({
            "No": index + 1,
            "Question": question,
            "Before_Answer (単純検索)": before_answer,
            "After_Answer (改善版/Excel)": after_answer
        })

    # 結果をExcelに保存
    result_df = pd.DataFrame(results)
    result_df.to_excel(OUTPUT_FILE, index=False)
    
    print(f"{'-'*60}")
    print(f"✨ 完了しました！結果ファイル: {OUTPUT_FILE}")
    print("このファイルを開いて、B列(Before)とC列(After)を見比べてください。")

if __name__ == "__main__":
    asyncio.run(main())