# services/llm.py
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from core.config import GEMINI_API_KEY
# from core.constants import SAFETY_SETTINGS # 今回は内部で定義して確実性を高めます

genai.configure(api_key=GEMINI_API_KEY)

# 安全設定を「ブロックなし」に設定し、誤検知による空レスポンスを防ぐ
ROBUST_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

class LLMService:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name

    # services/llm.py の get_embedding 関数を以下で完全に書き換えてください

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_embedding(self, text: str, model: str = "models/gemini-embedding-001") -> list[float]:
        """
        埋め込みベクトルを取得する。
        DBの仕様(768次元)に合わせて、強制的に次元を制限します。
        """
        try:
            # text-embedding-004 などの新しいモデルが選ばれた場合でも
            # 確実に 768 次元で出力されるように設定
            result = await genai.embed_content_async(
                model=model,
                content=text,
                task_type="retrieval_query",
                output_dimensionality=768  # ★ 768次元に固定
            )
            return result["embedding"]
        except Exception as e:
            # もし gemini-embedding-001 が output_dimensionality 未対応でエラーになった場合
            # 標準形式で再試行する
            result = await genai.embed_content_async(
                model=model,
                content=text,
                task_type="retrieval_query"
            )
            # 万が一 3072次元で返ってきた場合は、先頭 768要素だけを切り出す（最終手段）
            emb = result["embedding"]
            return emb[:768] if len(emb) > 768 else emb

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_stream(self, prompt: str, system_prompt: str = None):
        model = genai.GenerativeModel(self.model_name)
        full_inputs = [system_prompt, prompt] if system_prompt else [prompt]
        
        # 安全設定を適用してストリーム生成
        return await model.generate_content_async(
            full_inputs, 
            stream=True,
            safety_settings=ROBUST_SAFETY_SETTINGS
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_json(self, prompt: str, schema_class: any):
        model = genai.GenerativeModel(self.model_name)
        
        # レスポンスを一度受け取る
        response = await model.generate_content_async(
            prompt,
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                response_schema=schema_class
            ),
            safety_settings=ROBUST_SAFETY_SETTINGS
        )

        # 【重要】中身が空の場合は例外を投げて、@retryによる再試行をトリガーする
        # これにより、呼び出し元での .text アクセスエラー（Critical Pipeline Error）を防ぐ
        if not response.parts:
            # ブロック理由があればログに出す等の拡張も可能
            reason = "Unknown"
            if response.prompt_feedback:
                reason = response.prompt_feedback.block_reason
            
            raise ValueError(f"Gemini returned an empty response. Reason: {reason}")
            
        return response