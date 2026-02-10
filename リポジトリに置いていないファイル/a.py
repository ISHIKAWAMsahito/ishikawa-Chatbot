import asyncio
# 修正したファイルが正しく読み込まれるかチェック
from services.llm import LLMService 

async def check_dimension():
    llm = LLMService()
    print("🚀 テスト開始: ベクトル生成中...")
    
    # テスト実行
    emb = await llm.get_embedding("テスト")
    
    print(f"📊 生成された次元数: {len(emb)}")
    
    if len(emb) == 768:
        print("✅ 成功！修正は正しく反映されています。evaluate.py を実行してください。")
    else:
        print(f"❌ 失敗... まだ {len(emb)} 次元です。services/llm.py が正しく上書きされていません。")

if __name__ == "__main__":
    asyncio.run(check_dimension())