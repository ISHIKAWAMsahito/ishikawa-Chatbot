import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

try:
    from core.config import GEMINI_API_KEY
except ImportError:
    import os
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

ROBUST_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

class LLMService:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    # ★重要: モデルを text-embedding-004 (3072次元) に戻します
    async def get_embedding(self, text: str, model: str = "models/gemini-embedding-001") -> list[float]:
        """
        テキストをベクトル化します。（3072次元のまま出力）
        """
        try:
            result = await genai.embed_content_async(
                model=model,
                content=text,
                task_type="retrieval_query"
            )
            # ★カット処理（[:768]）を削除しました。そのまま返します。
            return result["embedding"]

        except Exception as e:
            logging.error(f"Embeddings generation failed: {e}")
            raise e

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_stream(self, prompt: str, system_prompt: str = None):
        model = genai.GenerativeModel(self.model_name)
        full_inputs = [system_prompt, prompt] if system_prompt else [prompt]
        return await model.generate_content_async(
            full_inputs, 
            stream=True,
            safety_settings=ROBUST_SAFETY_SETTINGS
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_json(self, prompt: str, schema_class: any):
        model = genai.GenerativeModel(self.model_name)
        response = await model.generate_content_async(
            prompt,
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                response_schema=schema_class
            ),
            safety_settings=ROBUST_SAFETY_SETTINGS
        )
        if not response.parts:
            raise ValueError("Gemini returned an empty response.")
        return response