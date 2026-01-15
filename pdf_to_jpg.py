import os
from pathlib import Path
from pdf2image import convert_from_path

# ---------------------------------------------------------
# ★ 設定: 今回のファイル用に書き換えました
# ---------------------------------------------------------

# 1. 変換したいPDFファイルのパス
TARGET_PDF = r"C:\dev\チャットボットdb\学事暦.pdf"

# 2. 画像を出力するフォルダ (新しいフォルダ名を指定)
OUTPUT_DIR = r"C:\dev\チャットボットdb\学事暦_jpg"

# 3. Popplerのbinフォルダのパス (変更なし)
POPPLER_PATH = r"C:\dev\poppler\poppler-25.11.0\Library\bin" 

# ---------------------------------------------------------

def convert_pdf():
    pdf_path = Path(TARGET_PDF)
    output_path = Path(OUTPUT_DIR)
    
    # 出力フォルダ作成
    output_path.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        print(f"❌ エラー: PDFが見つかりません: {pdf_path}")
        return

    print(f"🚀 変換開始: {pdf_path.name}")
    print(f"📂 出力先: {output_path}")
    
    try:
        # PDFを画像に変換 (300dpi)
        images = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
        
        print(f"📄 全 {len(images)} ページを保存します...")

        for i, image in enumerate(images):
            # ファイル名: 02_新札幌_学部共通事項_001.jpg
            save_name = f"{pdf_path.stem}_{i+1:03}.jpg"
            save_path = output_path / save_name
            
            image.save(save_path, "JPEG", quality=95)
            
            if (i + 1) % 5 == 0:
                print(f"  -> {i+1}ページ目まで保存完了")
            
    except Exception as e:
        print(f"❌ エラー: {e}")

    print("\n✅ --- 画像展開完了 ---")
    print(f"確認: {OUTPUT_DIR}")

if __name__ == "__main__":
    convert_pdf()