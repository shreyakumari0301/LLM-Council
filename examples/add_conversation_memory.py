"""
Example: How to add conversation history/memory

This enables multi-turn conversations.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field

@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[float] = None

class ConversationMemory:
    """Manages conversation history for context-aware responses."""
    
    def __init__(self, max_history: int = 10):
        self.history: List[Message] = []
        self.max_history = max_history
    
    def add_message(self, role: str, content: str):
        """Add a message to history."""
        import time
        message = Message(role=role, content=content, timestamp=time.time())
        self.history.append(message)
        
        # Keep only last N messages
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_context(self, max_tokens: int = 2000) -> List[Dict]:
        """Get recent messages as context for LLM."""
        context = []
        total_tokens = 0
        
        # Start from most recent and work backwards
        for message in reversed(self.history):
            # Rough token estimate (1 token ≈ 4 characters)
            msg_tokens = len(message.content) // 4
            if total_tokens + msg_tokens > max_tokens:
                break
            context.insert(0, {"role": message.role, "content": message.content})
            total_tokens += msg_tokens
        
        return context
    
    def build_prompt_with_context(self, new_prompt: str) -> str:
        """Build a prompt that includes conversation history."""
        context = self.get_context()
        
        # Format context
        context_text = "\n".join([
            f"{msg['role'].title()}: {msg['content']}"
            for msg in context
        ])
        
        if context_text:
            return f"Previous conversation:\n{context_text}\n\nUser: {new_prompt}\nAssistant:"
        return new_prompt
    
    def clear(self):
        """Clear conversation history."""
        self.history = []


# Usage in council.py:
# class LLMCouncil:
#     def __init__(self):
#         self.memory = ConversationMemory()
#     
#     async def consult_with_memory(self, prompt: str):
#         # Build prompt with context
#         contextual_prompt = self.memory.build_prompt_with_context(prompt)
#         
#         # Get response
#         result = await self.consult(contextual_prompt)
#         
#         # Save to memory
#         self.memory.add_message("user", prompt)
#         self.memory.add_message("assistant", result["final_answer"])
#         
#         return result

