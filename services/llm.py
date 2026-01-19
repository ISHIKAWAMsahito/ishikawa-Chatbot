# services/llm.py
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from core.config import GEMINI_API_KEY
from core.constants import SAFETY_SETTINGS

genai.configure(api_key=GEMINI_API_KEY)

class LLMService:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_embedding(self, text: str, model: str = "models/text-embedding-004") -> list[float]:
        """
        埋め込みベクトルを取得する
        :param text: テキスト
        :param model: 使用する埋め込みモデル名 (デフォルトはフォールバック用)
        """
        result = await genai.embed_content_async(
            model=model,  # 引数で渡されたモデルを使用
            content=text,
            task_type="retrieval_query"
        )
        return result["embedding"]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_stream(self, prompt: str, system_prompt: str = None):
        model = genai.GenerativeModel(self.model_name)
        full_inputs = [system_prompt, prompt] if system_prompt else [prompt]
        return await model.generate_content_async(
            full_inputs, 
            stream=True,
            safety_settings=SAFETY_SETTINGS
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_json(self, prompt: str, schema_class: any):
        model = genai.GenerativeModel(self.model_name)
        return await model.generate_content_async(
            prompt,
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                response_schema=schema_class
            ),
            safety_settings=SAFETY_SETTINGS
        )