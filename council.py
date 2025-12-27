import asyncio
import os
from typing import List, Dict, Optional
from providers import GroqProvider, OllamaProvider, MistralProvider, BaseLLMProvider

class LLMCouncil:
    """Council system that queries multiple LLMs and cross-verifies responses."""
    
    def __init__(self):
        self.providers: List[BaseLLMProvider] = []
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all LLM providers based on environment configuration."""
        # Groq - always try to initialize if API key is present
        try:
            self.providers.append(GroqProvider())
        except Exception as e:
            print(f"Warning: Could not initialize Groq: {e}")
        
        # Ollama - only initialize if USE_OLLAMA is true
        use_ollama = os.getenv("USE_OLLAMA", "false").lower() in ("true", "1", "yes")
        if use_ollama:
            try:
                self.providers.append(OllamaProvider())
            except Exception as e:
                print(f"Warning: Could not initialize Ollama: {e}")
        
        # Mistral - optional, only if API key is present and not using Ollama
        # (Since Mistral API is costly, we skip it if Ollama is enabled)
        if not use_ollama:
            try:
                self.providers.append(MistralProvider())
            except Exception as e:
                print(f"Warning: Could not initialize Mistral: {e}")
        
        if not self.providers:
            raise RuntimeError("No LLM providers could be initialized")
    
    async def query_all(self, prompt: str) -> Dict[str, str]:
        """Query all providers independently."""
        tasks = [provider.query(prompt) for provider in self.providers]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = {}
        for provider, response in zip(self.providers, responses):
            provider_name = provider.get_provider_name()
            if isinstance(response, Exception):
                results[provider_name] = f"Error: {str(response)}"
            else:
                results[provider_name] = response
        
        return results
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple similarity score between two texts."""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _cross_verify(self, responses: Dict[str, str]) -> Dict[str, float]:
        """Cross-verify responses and calculate confidence scores."""
        provider_names = list(responses.keys())
        confidence_scores = {name: 0.0 for name in provider_names}
        
        # Compare each response with all others
        for i, provider1 in enumerate(provider_names):
            for provider2 in provider_names[i+1:]:
                similarity = self._calculate_similarity(
                    responses[provider1],
                    responses[provider2]
                )
                confidence_scores[provider1] += similarity
                confidence_scores[provider2] += similarity
        
        # Normalize scores
        max_score = max(confidence_scores.values()) if confidence_scores.values() else 1.0
        if max_score > 0:
            confidence_scores = {
                name: score / max_score 
                for name, score in confidence_scores.items()
            }
        
        return confidence_scores
    
    def _select_optimal_answer(self, responses: Dict[str, str], confidence_scores: Dict[str, float]) -> tuple:
        """Select the most optimal answer based on confidence scores."""
        if not responses:
            return None, None
        
        # Find provider with highest confidence
        best_provider = max(confidence_scores.items(), key=lambda x: x[1])[0]
        best_answer = responses[best_provider]
        best_confidence = confidence_scores[best_provider]
        
        return best_answer, best_provider
    
    async def consult(self, prompt: str, verbose: bool = True) -> Dict:
        """Main council consultation method."""
        if verbose:
            print("🤖 Council members are deliberating...\n")
        
        # Step 1: Independent queries
        if verbose:
            print("📝 Step 1: Independent responses from each LLM:")
        responses = await self.query_all(prompt)
        
        if verbose:
            for provider, response in responses.items():
                print(f"\n{provider}:")
                print(f"  {response[:200]}..." if len(response) > 200 else f"  {response}")
        
        # Step 2: Cross-verification
        if verbose:
            print("\n\n🔍 Step 2: Cross-verifying responses...")
        confidence_scores = self._cross_verify(responses)
        
        if verbose:
            print("\nConfidence scores:")
            for provider, score in sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True):
                print(f"  {provider}: {score:.2%}")
        
        # Step 3: Select optimal answer
        if verbose:
            print("\n\n✅ Step 3: Optimal answer selected:")
        optimal_answer, best_provider = self._select_optimal_answer(responses, confidence_scores)
        
        result = {
            "question": prompt,
            "responses": responses,
            "confidence_scores": confidence_scores,
            "optimal_answer": optimal_answer,
            "selected_provider": best_provider,
            "confidence": confidence_scores.get(best_provider, 0.0)
        }
        
        if verbose:
            print(f"\nSelected Provider: {best_provider} (Confidence: {confidence_scores.get(best_provider, 0.0):.2%})")
            print(f"\nOptimal Answer:\n{optimal_answer}\n")
        
        return result

