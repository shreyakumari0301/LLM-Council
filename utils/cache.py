"""Response caching to save API costs."""
import hashlib
from typing import Optional

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

# Global cache instance
_cache = SimpleCache()

