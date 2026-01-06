from abc import ABC, abstractmethod
from typing import Optional

# Lazy import cache to avoid startup delays
_cache = None

def get_cache():
    """Lazy load cache to avoid import delays."""
    global _cache
    if _cache is None:
        try:
            from utils.cache import _cache as cache_instance
            _cache = cache_instance
        except ImportError:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from utils.cache import _cache as cache_instance
            _cache = cache_instance
    return _cache

class BaseLLMProvider(ABC):
    """Base class for all LLM providers."""
    
    async def query(self, prompt: str, model: Optional[str] = None) -> str:
        """Query the LLM and return the response (with caching)."""
        cache = get_cache()
        # Check cache first
        cached = cache.get(prompt, self.get_provider_name())
        if cached:
            return cached
        
        # Query LLM
        response = await self._query_llm(prompt, model)
        
        # Cache response
        cache.set(prompt, self.get_provider_name(), response)
        
        return response
    
    @abstractmethod
    async def _query_llm(self, prompt: str, model: Optional[str] = None) -> str:
        """Internal method to query the LLM (implemented by subclasses)."""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the provider."""
        pass

