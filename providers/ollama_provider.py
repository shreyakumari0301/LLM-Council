import os
from typing import Optional
import ollama
import requests
from .base import BaseLLMProvider

class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider implementation."""
    
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.default_model = model or os.getenv("OLLAMA_MODEL", "llama3.1")
        
        # Check if Ollama is running
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code != 200:
                raise ConnectionError(f"Ollama server not responding at {self.base_url}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Ollama is not running or not accessible at {self.base_url}. "
                f"Please start Ollama: 'ollama serve' or install it from https://ollama.ai/"
            ) from e
        
        # Set the base URL for ollama client
        try:
            if hasattr(ollama, 'Client'):
                self.client = ollama.Client(host=self.base_url)
            else:
                self.client = None
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Ollama client: {str(e)}")
    
    async def _query_llm(self, prompt: str, model: Optional[str] = None) -> str:
        """Query Ollama API."""
        import asyncio
        try:
            model = model or self.default_model
            
            # Check if model is available and normalize model name
            try:
                models_response = requests.get(f"{self.base_url}/api/tags", timeout=2)
                if models_response.status_code == 200:
                    available_models = [m.get("name", "") for m in models_response.json().get("models", [])]
                    # Normalize model name - remove tags for matching (e.g., "mistral:latest" -> "mistral")
                    model_base = model.split(":")[0]
                    
                    # Check if model exists (with or without tag)
                    model_found = False
                    actual_model = model
                    for avail_model in available_models:
                        if avail_model == model or avail_model.startswith(f"{model_base}:"):
                            actual_model = avail_model
                            model_found = True
                            break
                    
                    if not model_found:
                        return f"Error: Model '{model}' not found. Available models: {', '.join(available_models) if available_models else 'none'}. Run: ollama pull {model_base}"
                    
                    # Use the actual model name with tag
                    model = actual_model
            except:
                pass  # Skip model check if it fails
            
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

