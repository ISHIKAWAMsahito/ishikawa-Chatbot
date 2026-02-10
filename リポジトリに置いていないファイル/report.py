import pandas as pd
import os
import glob

# 一番新しいレポートファイルを自動で探す（ablation または deep_analysis）
list_of_files = glob.glob('ablation_report_*.xlsx') + glob.glob('deep_analysis_report_*.xlsx')
if not list_of_files:
    print("❌ レポートファイルが見つかりません。")
    exit()

# 最も新しいファイルを選択
LATEST_FILE = max(list_of_files, key=os.path.getctime)
OUTPUT_FILE = f"graphed_{LATEST_FILE}"

def create_excel_with_charts():
    print(f"📂 読み込み中: {LATEST_FILE}")
    df = pd.read_excel(LATEST_FILE)
    
    # Excel作成エンジンを起動
    writer = pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Result', index=False)
    
    workbook = writer.book
    worksheet = writer.sheets['Result']
    max_row = len(df) + 1
    
    # --- 1. 【検索力の向上】をグラフ化 ---
    chart1 = workbook.add_chart({'type': 'column'})
    
    # 寄与度分析(ablation)用: "検索向上数" がある場合
    if "検索向上数" in df.columns:
        col_idx = df.columns.get_loc("検索向上数")
        chart1.add_series({
            'name':       '検索ヒット向上数 (件)',
            'categories': ['Result', 1, 0, max_row, 0],
            'values':     ['Result', 1, col_idx, max_row, col_idx],
            'fill':       {'color': '#3498db'},
        })
        chart1.set_title({'name': 'クエリ拡張による検索結果の純増数'})
    
    # 深層分析(deep_analysis)用: "Before_ヒット数" がある場合
    elif "Before_ヒット数" in df.columns and "After_ヒット数" in df.columns:
        col_before = df.columns.get_loc("Before_ヒット数")
        col_after = df.columns.get_loc("After_ヒット_数") if "After_ヒット_数" in df.columns else df.columns.get_loc("After_ヒット数")
        
        chart1.add_series({
            'name': '改善前',
            'values': ['Result', 1, col_before, max_row, col_before],
            'fill': {'color': '#bdc3c7'},
        })
        chart1.add_series({
            'name': '改善後',
            'values': ['Result', 1, col_after, max_row, col_after],
            'fill': {'color': '#3498db'},
        })
        chart1.set_title({'name': '検索ヒット数の比較'})

    chart1.set_x_axis({'name': '質問番号'})
    chart1.set_y_axis({'name': '件数'})
    worksheet.insert_chart('K2', chart1)

    # --- 2. 【改善率または品質】をグラフ化 ---
    chart2 = workbook.add_chart({'type': 'line'})
    
    # ablation用: "改善率(%)"
    if "改善率(%)" in df.columns:
        col_idx = df.columns.get_loc("改善率(%)")
        chart2.add_series({
            'name': 'スコア改善率 (%)',
            'categories': ['Result', 1, 0, max_row, 0],
            'values': ['Result', 1, col_idx, max_row, col_idx],
            'line': {'color': '#e67e22', 'width': 2},
            'marker': {'type': 'circle', 'size': 5},
        })
        chart2.set_title({'name': '旧方式に対する精度向上率'})
        chart2.set_y_axis({'name': 'パーセント (%)'})

    # deep_analysis用: "After_Rerank最高点"
    elif "After_Rerank最高点" in df.columns:
        col_idx = df.columns.get_loc("After_Rerank最高点")
        chart2.add_series({
            'name': 'AI採点 (Rerank)',
            'categories': ['Result', 1, 0, max_row, 0],
            'values': ['Result', 1, col_idx, max_row, col_idx],
            'line': {'color': '#2ecc71', 'width': 2},
            'marker': {'type': 'square', 'size': 5},
        })
        chart2.set_title({'name': '回答の確信度スコア'})
        chart2.set_y_axis({'name': '点数 (10点満点)', 'min': 0, 'max': 10})

    worksheet.insert_chart('K18', chart2)

    writer.close()
    print(f"✨ グラフ付きExcelを作成しました: {OUTPUT_FILE}")

if __name__ == "__main__":
    create_excel_with_charts()