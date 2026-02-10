import pandas as pd
from langsmith import Client
from dotenv import load_dotenv

# 1. 環境設定
load_dotenv(r"C:\dev\ishikawa-Chatbot\ishikawa-Chatbot.env")

# ==========================================
# 比較したい2つのプロジェクト名を設定
# ==========================================
PROJECT_A = "SGU_Evaluation_10_Questions_v2"  # 比較対象A (例: 改善前)
PROJECT_B = "ishikawa-chatbot-eval"           # 比較対象B (例: 改善後)

def fetch_project_data(project_name):
    """指定されたプロジェクトから実行結果を取得してDataFrameにする関数"""
    print(f"📥 '{project_name}' からデータを取得中...")
    client = Client()
    
    # プロジェクトの実行履歴を取得 (親Runのみ)
    runs = list(client.list_runs(
        project_name=project_name,
        execution_order=1,
        error=False
    ))
    
    data = []
    for run in runs:
        # 入力(質問)の取得
        inputs = run.inputs or {}
        question = inputs.get("question") or inputs.get("input") or str(inputs)
        
        # 出力(回答)の取得
        outputs = run.outputs or {}
        answer = outputs.get("answer") or outputs.get("output") or str(outputs)
        
        # 評価スコア(accuracy)の取得
        feedbacks = list(client.list_feedback(run_ids=[run.id]))
        score = None
        for f in feedbacks:
            if f.key == "accuracy":
                score = f.score
                break
        
        # レイテンシ(秒)
        latency = (run.end_time - run.start_time).total_seconds() if run.end_time else 0
        
        data.append({
            "Question": question.strip(), # マージ用に空白除去
            "Answer": answer,
            "Score": score,
            "Latency": latency
        })
        
    return pd.DataFrame(data)

def main():
    # 1. 両方のデータを取得
    df_a = fetch_project_data(PROJECT_A)
    df_b = fetch_project_data(PROJECT_B)

    if df_a.empty or df_b.empty:
        print("❌ どちらかのデータ取得に失敗しました。プロジェクト名を確認してください。")
        return

    print(f"\n🔄 データを結合して比較表を作成します...")

    # 2. 「質問(Question)」をキーにしてデータを結合 (inner join または outer join)
    # 接尾辞 (_A, _B) をつけて区別します
    merged_df = pd.merge(
        df_a, 
        df_b, 
        on="Question", 
        how="outer", 
        suffixes=(f'_{PROJECT_A}', f'_{PROJECT_B}')
    )

    # 3. 差分（Delta）を計算
    # スコアの変化 (A - B) ※Aが改善後なら、プラスが良い
    score_col_a = f"Score_{PROJECT_A}"
    score_col_b = f"Score_{PROJECT_B}"
    
    # Noneを0として扱うか、NaNのままにするかは分析方針次第ですが、ここでは計算用にfillna(0)せずそのまま計算
    merged_df["Score_Diff"] = merged_df[score_col_a] - merged_df[score_col_b]
    
    # 時間の変化 (A - B)
    lat_col_a = f"Latency_{PROJECT_A}"
    lat_col_b = f"Latency_{PROJECT_B}"
    merged_df["Latency_Diff"] = merged_df[lat_col_a] - merged_df[lat_col_b]

    # 4. 見やすいように列を並べ替え
    columns_order = [
        "Question", 
        score_col_a, score_col_b, "Score_Diff",  # スコア比較
        lat_col_a, lat_col_b, "Latency_Diff",    # 時間比較
        f"Answer_{PROJECT_A}", f"Answer_{PROJECT_B}" # 回答比較
    ]
    
    # 存在しない列がある場合のエラー回避
    final_cols = [c for c in columns_order if c in merged_df.columns]
    result_df = merged_df[final_cols]

    # 5. 結果の表示と保存
    print("\n" + "="*60)
    print(f"📊 比較レポート: {PROJECT_A} vs {PROJECT_B}")
    print("="*60)
    
    avg_a = df_a["Score"].mean()
    avg_b = df_b["Score"].mean()
    print(f"🔹 平均スコア ({PROJECT_A}): {avg_a:.3f}")
    print(f"🔹 平均スコア ({PROJECT_B}): {avg_b:.3f}")
    print(f"📈 スコア改善幅: {avg_a - avg_b:+.3f}")
    print("-" * 30)
    
    lat_a = df_a["Latency"].mean()
    lat_b = df_b["Latency"].mean()
    print(f"🔹 平均時間 ({PROJECT_A}): {lat_a:.2f}s")
    print(f"🔹 平均時間 ({PROJECT_B}): {lat_b:.2f}s")
    print(f"⏱ 時間の増減: {lat_a - lat_b:+.2f}s")
    print("="*60)

    output_file = "comparison_result.csv"
    result_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n✅ 詳細な比較データを '{output_file}' に保存しました。")
    print("   Excelで開き、「Score_Diff」がプラスになっている質問を確認してください。")

if __name__ == "__main__":
    main()