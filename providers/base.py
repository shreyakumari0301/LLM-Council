from abc import ABC, abstractmethod
from typing import Optional

class BaseLLMProvider(ABC):
    """Base class for all LLM providers."""
    
    @abstractmethod
    async def query(self, prompt: str, model: Optional[str] = None) -> str:
        """Query the LLM and return the response."""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the provider."""
        pass

