import os
import logging
from typing import Dict, Optional
from dataclasses import dataclass
import google.generativeai as genai
import requests


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
                content=getattr(response, 'content', ''),
                usage_stats=getattr(response, 'usage_metadata', None),
                model_used=model_name,
                finish_reason=getattr(response, 'finish_reason', 'STOP')
            )

        except Exception as e:
            logging.error(f"Gemini API call failed: {e}")
            raise RuntimeError(f"Failed to call Gemini API: {e}")


class PurdueLLMManager:
    def __init__(self):

        self.api_key = os.getenv("GEN_AI_STUDIO_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEN_AI_STUDIO_API_KEY not configured. Cannot make API calls."
            )

    def call_genai_api(self, prompt: str, model: Optional[str] = None
                       ) -> LLMResponse:

        url = "https://genai.rcac.purdue.edu/api/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": model or "llama3.1:latest",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise JSON generator. You ALWAYS "
                               "respond with valid JSON and nothing else. No "
                               "explanations, no markdown, no code blocks."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 512,
            "stream": False
        }
        
        try:
            response = requests.post(url, headers=headers, json=body)
            if response.status_code == 200:
                # Parse the JSON response
                response_data = response.json()
                
                # Extract the content from the response
                choices = response_data.get("choices", [])
                if choices and len(choices) > 0:
                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                else:
                    content = ""
                
                # Extract usage stats if available
                usage = response_data.get("usage", {})
                
                # Get finish reason
                finish_reason = "STOP"
                if choices:
                    finish_reason = choices[0].get("finish_reason", "STOP")
                
                return LLMResponse(
                    content=content,
                    usage_stats=usage if usage else None,
                    model_used=model or "llama3.1:latest",
                    finish_reason=finish_reason
                )
            else:
                raise Exception(f"Error: {response.status_code}, "
                                f"{response.text}")
        except Exception as e:
            logging.error(f"Purdue LLM API call failed: {e}")
            raise RuntimeError(f"Failed to call Purdue LLM API: {e}")
