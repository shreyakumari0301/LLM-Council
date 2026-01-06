"""
Example: How to add response caching

This saves API costs by caching responses.
"""

import hashlib
import json
import os
from typing import Optional
from functools import lru_cache

class SimpleCache:
    """Simple in-memory cache for responses."""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
    
    def _get_key(self, prompt: str, provider: str) -> str:
        """Generate cache key from prompt and provider."""
        content = f"{provider}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, prompt: str, provider: str) -> Optional[str]:
        """Get cached response if exists."""
        key = self._get_key(prompt, provider)
        return self.cache.get(key)
    
    def set(self, prompt: str, provider: str, response: str):
        """Cache a response."""
        key = self._get_key(prompt, provider)
        
        # Evict oldest if cache is full
        if len(self.cache) >= self.max_size:
            # Remove first item (FIFO)
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        
        self.cache[key] = response
    
    def clear(self):
        """Clear all cached responses."""
        self.cache.clear()


# Usage in BaseLLMProvider:
# class BaseLLMProvider(ABC):
#     _cache = SimpleCache()
#     
#     async def query(self, prompt: str, model: Optional[str] = None) -> str:
#         # Check cache first
#         cached = self._cache.get(prompt, self.get_provider_name())
#         if cached:
#             return cached
#         
#         # Query LLM
#         response = await self._query_llm(prompt, model)
#         
#         # Cache response
#         self._cache.set(prompt, self.get_provider_name(), response)
#         
#         return response

