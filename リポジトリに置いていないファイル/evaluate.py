import asyncio
import os
import pandas as pd
import re
from dotenv import load_dotenv

# 1. 環境設定
env_path = r"C:\dev\ishikawa-Chatbot\ishikawa-Chatbot.env"
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

from langsmith import Client, aevaluate
from langsmith.schemas import Run, Example

try:
    from core.database import SupabaseClientManager 
    from services.llm import LLMService
    from services.search import SearchService
    from core.constants import PARAMS 
    from services import prompts 
except ImportError as e:
    print(f"❌ インポートエラー: {e}")
    exit(1)

INPUT_FILE = "質問とチャットボットシステムの回答集.xlsx"
DATASET_NAME = "SGU_Evaluation_Clean_v1"  # さっき作った新しいデータセット
NUM_TEST_QUESTIONS = 10 

# ==========================================
# 1. チャットボット処理 (ハイブリッド + Rerank + LitM)
# ==========================================
async def chatbot_pipeline(inputs: dict, config: str) -> dict:
    llm = LLMService()
    db = SupabaseClientManager(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
    search = SearchService(llm)

    question = inputs.get("question")
    if not question: return {"answer": "質問なし"}

    # configが 'after' なので use_full は True になります
    use_full = (config == 'after')
    
    try:
        # 1. クエリ拡張 (Afterのみ)
        query = await search.expand_query(question) if use_full else question
        
        # 2. 検索 (取得数を20→10に減らしてAPI節約)
        emb = await llm.get_embedding(query)
        raw_docs = db.client.rpc("match_documents_hybrid", {
            "p_collection_name": "student-knowledge-base",
            "p_query_text": query,
            "p_query_embedding": emb,
            "p_match_count": 10  # 【修正】API制限回避のため20から10へ削減
        }).execute().data or []

        # 3. 加工処理
        if use_full and raw_docs:
            # Afterモード: 重複カット、Rerank、配置変更を実行
            docs = search.filter_diversity(raw_docs)
            docs = await search.rerank(question, docs, top_k=5)
            docs = search.reorder_litm(docs)
        else:
            # Beforeモード: 上位5件をそのまま取得
            docs = raw_docs[:5]

        # 4. 生成
        context_str = "\n\n".join([d.get('content','') for d in docs])
        prompt = f"質問: {question}\n\n<context>\n{context_str}\n</context>"
        
        res_stream = await llm.generate_stream(prompt, prompts.SYSTEM_GENERATION)
        answer = ""
        async for chunk in res_stream:
            if chunk.text: answer += chunk.text
        
        return {
            "answer": answer.strip(),
            "contexts": [d.get('content','') for d in docs] 
        }
        
    except Exception as e:
        return {"answer": f"Error: {e}"}

# ==========================================
# 2. 自動採点
# ==========================================
async def quality_evaluator(run: Run, example: Example) -> dict:
    llm = LLMService()
    student_ans = run.outputs.get("answer", "")
    question = example.inputs.get("question", "")
    ground_truth = example.outputs.get("answer", "")

    # 正解データがない場合は採点スキップ（エラー防止）
    if not ground_truth:
        return {"key": "accuracy", "score": 0.0}

    prompt = f"""
    [質問]: {question}
    [正解]: {ground_truth}
    [回答]: {student_ans}
    
    上記を0-10点で採点し、最後に "Score: 数値" と書いてください。
    """
    try:
        res = await llm.generate_stream(prompt)
        text = ""
        async for chunk in res:
            if chunk.text: text += chunk.text
        
        match = re.search(r'Score:\s*(\d+)', text)
        score = int(match.group(1)) if match else 0
        return {"key": "accuracy", "score": score / 10.0}
    except:
        return {"key": "accuracy", "score": 0.0}

# ==========================================
# 3. メイン処理
# ==========================================
async def main():
    print(f"📂 ファイル読み込み: {INPUT_FILE}")
    ls_client = Client()
    
    # データセットの確認（なければ作るが、基本はあるはず）
    if not ls_client.has_dataset(dataset_name=DATASET_NAME):
        print(f"📦 データセット作成中...")
        df = pd.read_excel(INPUT_FILE).head(NUM_TEST_QUESTIONS)
        ds = ls_client.create_dataset(dataset_name=DATASET_NAME)
        for _, row in df.iterrows():
            ls_client.create_example(
                inputs={"question": str(row["Question"])},
                outputs={"answer": str(row.get("Answer", ""))},
                dataset_id=ds.id
            )
    else:
        print(f"✅ 既存のデータセット '{DATASET_NAME}' を使用します。")
    
    # 【重要】モードを 'after' (改善版) に設定
    for mode in ['before']:
        print(f"\n🚀 実験モード: [{mode.upper()}] (Rerank & LitM) を実行中...")
        
        async def target_wrapper(inputs):
            # 【修正】APIエラー回避のため待機時間を延長 (15秒)
            print("⏳ API制限回避のため 15秒 待機中...")
            await asyncio.sleep(15) 
            return await chatbot_pipeline(inputs, mode)

        try:
            await aevaluate(
                target_wrapper, 
                data=DATASET_NAME, 
                evaluators=[quality_evaluator],
                # 【重要】実験名を変更して Baseline と区別する
                experiment_prefix=f"After_Rerank_LitM", 
                max_concurrency=1
            )
        except Exception as e:
            print(f"❌ 評価エラー ({mode}): {e}")

    print("\n✨ 改善版の計測完了！LangSmithで Baseline と比較してください。")

if __name__ == "__main__":
    asyncio.run(main())