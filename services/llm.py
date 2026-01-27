import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from langsmith import traceable # LangSmith追跡用

try:
    from core.config import GEMINI_API_KEY
except ImportError:
    import os
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

# 安全設定: 学術・事務的な回答を阻害しないようブロックを解除
ROBUST_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

class LLMService:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name

    @traceable(name="Gemini_Embedding", run_type="embedding")
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_embedding(self, text: str, model: str = "models/gemini-embedding-001") -> list[float]:
        """
        テキストをベクトル化します。
        LangSmith上では 'embedding' タイプとして記録され、コスト分析などがしやすくなります。
        """
        try:
            result = await genai.embed_content_async(
                model=model,
                content=text,
                task_type="retrieval_query"
            )
            return result["embedding"]

        except Exception as e:
            logging.error(f"Embeddings generation failed: {e}")
            raise e

    @traceable(name="Gemini_Generation_Stream", run_type="llm")
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_stream(self, prompt: str, system_prompt: str = None):
        """
        ストリーム生成を行います。
        LangSmith上ではプロンプトと生成された回答が対になって表示されます。
        """
        model = genai.GenerativeModel(self.model_name)
        full_inputs = [system_prompt, prompt] if system_prompt else [prompt]
        return await model.generate_content_async(
            full_inputs, 
            stream=True,
            safety_settings=ROBUST_SAFETY_SETTINGS
        )

    @traceable(name="Gemini_Generation_JSON", run_type="llm")
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_json(self, prompt: str, schema_class: any):
        """
        JSONモードでの生成を行います（主にリランク用）。
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
        if not response.parts:
            raise ValueError("Gemini returned an empty response.")
        return response