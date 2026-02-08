import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from typing import List, Dict, Any, AsyncGenerator
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
    def __init__(self, model_name: str = "models/gemini-2.5-flash"):
        self.model_name = model_name

    @traceable(name="Gemini_Embedding", run_type="embedding")
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_embedding(self, text: str, model: str = "models/gemini-embedding-001") -> list[float]:
        """
        テキストをベクトル化します。
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
        汎用的なストリーム生成を行います（クエリ拡張や分析用）。
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
        JSONモードでの生成を行います（リランク用）。
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

    # ★追加: このメソッドが不足していたためエラーになっていました
    @traceable(name="Gemini_Chat_Response_Stream", run_type="llm")
    async def generate_response_stream(
        self, 
        query: str, 
        context_docs: List[Dict], 
        history: List[Dict], 
        system_prompt: str
    ) -> AsyncGenerator[str, None]:
        """
        メインチャット用レスポンス生成メソッド
        (RAGコンテキストと履歴を考慮して回答を生成)
        """
        try:
            # 1. モデルの初期化 (システムプロンプトを設定)
            model = genai.GenerativeModel(
                self.model_name,
                system_instruction=system_prompt
            )

            # 2. 履歴データの変換 (Gemini形式へ)
            gemini_history = []
            for msg in history:
                role = "user" if msg["role"] == "user" else "model"
                content = msg.get("content", "")
                if content: # 空のメッセージは除外
                    gemini_history.append({"role": role, "parts": [content]})

            # 3. チャットセッション開始
            chat = model.start_chat(history=gemini_history)

            # 4. 回答生成 (ストリーミング)
            # safety_settingsも適用して堅牢にする
            response = await chat.send_message_async(
                query, 
                stream=True,
                safety_settings=ROBUST_SAFETY_SETTINGS
            )

            async for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logging.error(f"Chat Generation Error: {e}")
            yield f"申し訳ありません。エラーが発生しました: {str(e)}"