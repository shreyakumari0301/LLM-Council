import os
from typing import Optional
from groq import Groq
from .base import BaseLLMProvider

class GroqProvider(BaseLLMProvider):
    """Groq LLM provider implementation."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.client = Groq(api_key=self.api_key)
        self.default_model = model or os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
    
    async def query(self, prompt: str, model: Optional[str] = None) -> str:
        """Query Groq API."""
        import asyncio
        try:
            model = model or self.default_model
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                )
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error from Groq: {str(e)}"
    
    def get_provider_name(self) -> str:
        return "Groq"

