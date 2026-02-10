import pandas as pd
import asyncio
from langsmith import Client, aevaluate
from dotenv import load_dotenv

# 環境設定
load_dotenv(r"C:\dev\ishikawa-Chatbot\ishikawa-Chatbot.env")

# 設定
NEW_DATASET_NAME = "SGU_Evaluation_Clean_v1" # 新しいクリーンな名前
INPUT_FILE = "質問とチャットボットシステムの回答集.xlsx"

async def create_clean_dataset_and_upload():
    client = Client()
    print(f"📂 Excel読み込み中: {INPUT_FILE}")

    # 1. Excelデータの準備
    try:
        df = pd.read_excel(INPUT_FILE)
        df.columns = [c.strip() for c in df.columns] # 列名のゴミ取り
    except Exception as e:
        print(f"❌ Excel読み込みエラー: {e}")
        return

    # 必要な列があるか確認
    if "Question" not in df.columns or "Answer" not in df.columns:
        print("❌ エラー: Excelに 'Question' または 'Answer' 列がありません。")
        return

    # 2. 新しいデータセットを作成（既に同名があればスキップ）
    if client.has_dataset(dataset_name=NEW_DATASET_NAME):
        print(f"⚠️ データセット '{NEW_DATASET_NAME}' は既に存在します。既存のものを使用します。")
    else:
        print(f"🆕 新しいデータセット '{NEW_DATASET_NAME}' を作成中...")
        dataset = client.create_dataset(dataset_name=NEW_DATASET_NAME)
        
        # 質問と正解(Ground Truth)を登録
        for q, a in zip(df['Question'], df['Answer']):
            client.create_example(
                inputs={"question": str(q).strip()},
                outputs={"answer": str(a).strip()}, # これが「理想の正解」になる
                dataset_id=dataset.id
            )
        print("✅ データセット作成完了！ゴミデータはもうありません。")

    # 3. 辞書化（マッチング用）
    qa_pairs = {}
    for q, a in zip(df['Question'], df['Answer']):
        clean_q = str(q).strip()
        qa_pairs[clean_q] = str(a).strip()

    # 4. モックシステム（Excelの回答を「システム回答」として返す）
    async def mock_system(inputs: dict):
        q = inputs.get("question")
        # 完全に一致するはず（データセット自体をExcelから作ったので）
        return {"answer": qa_pairs.get(q, "Error: データ不一致")}

    print(f"🚀 'Answer'列を {NEW_DATASET_NAME} の実験結果として登録中...")

    # 5. 評価実行（採点AIなしで、まずは登録だけ行う）
    # ※自動採点を入れたい場合は evaluators=[...] を追加してください
    await aevaluate(
        mock_system,
        data=NEW_DATASET_NAME,
        experiment_prefix="Production_Result_Fixed",
        max_concurrency=1
    )

    print(f"\n✨ 完了しました！")
    print(f"LangSmithで '{NEW_DATASET_NAME}' を開き、'Production_Result_Fixed' を確認してください。")

if __name__ == "__main__":
    asyncio.run(create_clean_dataset_and_upload())