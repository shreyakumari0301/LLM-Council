import os
from typing import Optional
import ollama
from .base import BaseLLMProvider

class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider implementation."""
    
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.default_model = model or os.getenv("OLLAMA_MODEL", "llama3.1")
        # Set the base URL for ollama client
        try:
            if hasattr(ollama, 'Client'):
                self.client = ollama.Client(host=self.base_url)
            else:
                self.client = None
        except Exception:
            self.client = None
    
    async def query(self, prompt: str, model: Optional[str] = None) -> str:
        """Query Ollama API."""
        import asyncio
        try:
            model = model or self.default_model
            loop = asyncio.get_event_loop()
            
            if self.client:
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.chat(
                        model=model,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                    )
                )
                return response['message']['content']
            else:
                # Fallback for older ollama versions
                response = await loop.run_in_executor(
                    None,
                    lambda: ollama.chat(
                        model=model,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                    )
                )
                return response['message']['content']
        except Exception as e:
            return f"Error from Ollama: {str(e)}"
    
    def get_provider_name(self) -> str:
        return "Ollama"

