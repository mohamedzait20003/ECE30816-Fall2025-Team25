import os
import logging
from typing import Dict, Optional
from dataclasses import dataclass
import google.generativeai as genai


@dataclass
class LLMResponse:
    content: str
    usage_stats: Optional[Dict] = None
    model_used: Optional[str] = None
    finish_reason: Optional[str] = None


class LLMManager:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not configured. Cannot make API calls."
            )

        genai.configure(
            api_key=self.api_key,
            transport="rest"
        )

        self.generation_cfg = genai.GenerationConfig(
            temperature=0.7,
            max_output_tokens=8192,
            top_p=0.95,
            top_k=40
        )

        self.default_model = "gemini-1.5-flash"
        self.max_tokens = 8192
        self.temperature = 0.7

    def call_gemini_api(self, prompt: str, model: Optional[str] = None
                        ) -> LLMResponse:

        model_name = model or self.default_model

        try:
            model_instance = genai.GenerativeModel(model_name)

            response = model_instance.generate_content(
                prompt,
                generation_config=self.generation_cfg
            )

            return LLMResponse(
                content=response.text,
                usage_stats=getattr(response, 'usage_metadata', None),
                model_used=model_name,
                finish_reason=getattr(response, 'finish_reason', 'STOP')
            )

        except Exception as e:
            logging.error(f"Gemini API call failed: {e}")
            raise RuntimeError(f"Failed to call Gemini API: {e}")
