import asyncio
import os
from typing import List, Dict
from providers import GroqProvider, OllamaProvider, MistralProvider, BaseLLMProvider

class LLMCouncil:
    """Mini LLM Council: multiple LLMs reason independently, critique each other, and synthesize a final answer."""

    def __init__(self):
        self.providers: List[BaseLLMProvider] = []
        self._initialize_providers()
        self._current_prompt: str = ""

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

    async def _generate_critiques(self, responses: Dict[str, str]) -> Dict[str, str]:
        """Each LLM critiques other LLM responses."""
        critiques = {}

        for provider in self.providers:
            provider_name = provider.get_provider_name()
            # Prepare other responses
            other_responses = {
                k: v for k, v in responses.items() if k != provider_name
            }

            critique_prompt = f"""
You are reviewing answers from other AI models.

Original Question:
{self._current_prompt}

Other Model Answers:
{other_responses}

Critique them by:
1. Identifying correct points
2. Pointing out missing details
3. Highlighting errors or weak reasoning
4. Suggesting improvements

Be concise and analytical.
"""
            critique = await provider.query(critique_prompt)
            critiques[provider_name] = critique

        return critiques

    async def _synthesize_answer(
        self,
        responses: Dict[str, str],
        critiques: Dict[str, str]
    ) -> str:
        """Synthesize a final answer from responses and critiques."""
        # Use the first provider (Groq) as synthesizer
        synthesizer = self.providers[0]

        synthesis_prompt = f"""
You are an expert AI mediator.

Original Question:
{self._current_prompt}

Initial Answers:
{responses}

Critiques from Models:
{critiques}

Task:
- Merge the strongest reasoning from all answers
- Correct any mistakes
- Fill missing gaps
- Produce ONE clear, accurate final answer

Do NOT mention models, critiques, or deliberation.
Just give the final answer.
"""
        final_answer = await synthesizer.query(synthesis_prompt)
        return final_answer

    async def consult(self, prompt: str, verbose: bool = True) -> Dict:
        """Main council consultation method."""
        self._current_prompt = prompt

        if verbose:
            print("🤖 Council members are deliberating...\n")

        # Step 1: Independent responses
        responses = await self.query_all(prompt)

        if verbose:
            print("📝 Independent Responses:")
            for provider, response in responses.items():
                print(f"\n{provider}:\n{response[:300]}")  # Truncate for display

        # Step 2: Cross-critiques
        if verbose:
            print("\n🔍 Cross-critiques:")
        critiques = await self._generate_critiques(responses)

        if verbose:
            for provider, critique in critiques.items():
                print(f"\n{provider} critique:\n{critique[:300]}")

        # Step 3: Synthesis
        if verbose:
            print("\n🧠 Synthesizing final answer...\n")
        final_answer = await self._synthesize_answer(responses, critiques)

        if verbose:
            print("✅ Final Answer:\n")
            print(final_answer)

        return {
            "question": prompt,
            "responses": responses,
            "critiques": critiques,
            "final_answer": final_answer
        }
