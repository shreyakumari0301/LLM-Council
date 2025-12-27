import os
from typing import Optional
from mistralai import Mistral
from .base import BaseLLMProvider

class MistralProvider(BaseLLMProvider):
    """Mistral LLM provider implementation."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "mistral-large-latest"):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
        self.client = Mistral(api_key=self.api_key)
        self.default_model = model
    
    async def query(self, prompt: str, model: Optional[str] = None) -> str:
        """Query Mistral API."""
        import asyncio
        try:
            model = model or self.default_model
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.complete(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                )
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error from Mistral: {str(e)}"
    
    def get_provider_name(self) -> str:
        return "Mistral"

