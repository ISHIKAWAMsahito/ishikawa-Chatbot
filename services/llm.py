import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

# プロジェクト内の設定ファイルを読み込み
# ※ もし core.config が見つからない場合は、直接 APIキー文字列を入れても動きますが、通常はこのまま使用します
try:
    from core.config import GEMINI_API_KEY
except ImportError:
    # 万が一読み込めない場合の安全策（本来は .env から読み込まれます）
    import os
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini APIの設定
genai.configure(api_key=GEMINI_API_KEY)

# 安全設定: 誤検知による空レスポンス（ハルシネーション扱い）を防ぐため、ブロックを無効化
ROBUST_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

class LLMService:
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """
        LLMサービスの初期化
        :param model_name: 生成に使用するモデル名
        """
        self.model_name = model_name

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_embedding(self, text: str, model: str = "models/gemini-embedding-001") -> list[float]:
        """
        テキストをベクトル化（埋め込み）します。
        【重要】データベース（Supabase）の定義に合わせて、強制的に768次元で出力します。
        """
        try:
            # API呼び出し
            # task_type="retrieval_query" は検索クエリ用の精度を高めます
            result = await genai.embed_content_async(
                model=model,
                content=text,
                task_type="retrieval_query"
            )
            
            # ベクトルを取り出す
            embedding = result["embedding"]
            
            # --- 次元の不一致（3072 vs 768）を防ぐための絶対的なガード処理 ---
            # text-embedding-004 などが 3072次元 を返してきた場合でも、
            # 先頭の 768次元 だけを切り出してデータベースに適合させます。
            if len(embedding) > 768:
                return embedding[:768]
            
            return embedding

        except Exception as e:
            logging.error(f"Embeddings generation failed: {e}")
            raise e

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_stream(self, prompt: str, system_prompt: str = None):
        """
        回答をストリーミング生成します。
        """
        model = genai.GenerativeModel(self.model_name)
        
        # システムプロンプトがある場合はリストの先頭に追加
        full_inputs = [system_prompt, prompt] if system_prompt else [prompt]
        
        return await model.generate_content_async(
            full_inputs, 
            stream=True,
            safety_settings=ROBUST_SAFETY_SETTINGS
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_json(self, prompt: str, schema_class: any):
        """
        JSON形式で構造化されたデータを生成します（Rerankやクエリ拡張で使用）。
        """
        model = genai.GenerativeModel(self.model_name)
        
        response = await model.generate_content_async(
            prompt,
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                response_schema=schema_class
            ),
            safety_settings=ROBUST_SAFETY_SETTINGS
        )

        # 空のレスポンスチェック
        if not response.parts:
            reason = "Unknown"
            if response.prompt_feedback:
                reason = response.prompt_feedback.block_reason
            raise ValueError(f"Gemini returned an empty response. Block Reason: {reason}")
            
        return response