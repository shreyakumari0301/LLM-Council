import asyncio
import os
from typing import List, Dict
from providers import GroqProvider, OllamaProvider, MistralProvider, BaseLLMProvider

# Timeout for API calls (seconds)
API_TIMEOUT = 30

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
        summary_prompt = f"""Provide a crisp, to-the-point answer. Be concise and direct - no fluff, just essential information.

Question: {prompt}

Answer:"""
        tasks = [
            asyncio.wait_for(provider.query(summary_prompt), timeout=API_TIMEOUT)
            for provider in self.providers
        ]
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
Review these answers for conciseness and accuracy.

Question: {self._current_prompt}

Other Answers: {other_responses}

Critique:
1. What's correct but too verbose?
2. What's missing or wrong?
3. How to make them more crisp and to-the-point?

Be brief in your critique.
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
You are an expert summarizer. Provide a crisp, to-the-point answer.

Question: {self._current_prompt}

Answers from models:
{responses}

Critiques: {critiques}

Task: Create ONE concise, direct answer that:
- Captures the essential information
- Removes redundancy and fluff
- Is crisp and to-the-point
- Corrects any errors

Be ruthless - keep it short and direct. No unnecessary words.
"""
        final_answer = await synthesizer.query(synthesis_prompt)
        return final_answer

    async def sequential_refine(self, prompt: str, verbose: bool = True) -> Dict:
        """Sequential refinement: LLM 1 generates, LLM 2+ refines iteratively."""
        self._current_prompt = prompt
        
        if len(self.providers) < 2:
            raise ValueError("Sequential refinement requires at least 2 providers")
        
        if verbose:
            print("🔄 Starting sequential refinement...\n")
        
        # Step 1: First LLM generates initial response
        first_provider = self.providers[0]
        summary_prompt = f"""Provide a crisp, to-the-point answer. Be concise and direct - no fluff, just the essential information.

Question: {prompt}

Answer:"""
        current_response = await asyncio.wait_for(
            first_provider.query(summary_prompt),
            timeout=API_TIMEOUT
        )
        
        if verbose:
            print(f"📝 Initial response from {first_provider.get_provider_name()}:")
            print(f"{current_response[:300]}...\n")
        
        refinement_chain = [{
            "provider": first_provider.get_provider_name(),
            "response": current_response,
            "stage": "initial"
        }]
        
        # Step 2: Each subsequent LLM analyzes and refines in one call
        for i, refiner in enumerate(self.providers[1:], 1):
            # Combined: Analyze and refine in a single API call
            refine_prompt = f"""Make this response more crisp and to-the-point. Remove unnecessary words, keep only essential information.

Question: {prompt}

Previous Response: {current_response}

Task: 
1. Identify what can be removed or condensed (be ruthless - cut fluff)
2. Provide a more concise, direct version

Format:
ANALYSIS:
[What to remove/condense]

IMPROVED RESPONSE:
[Crisp, to-the-point version - shorter than previous]"""
            
            combined_response = await asyncio.wait_for(
                refiner.query(refine_prompt),
                timeout=API_TIMEOUT
            )
            
            # Parse the response to extract analysis and refined output
            if "ANALYSIS:" in combined_response and "IMPROVED RESPONSE:" in combined_response:
                parts = combined_response.split("IMPROVED RESPONSE:")
                analysis = parts[0].replace("ANALYSIS:", "").strip()
                refined_response = parts[1].strip() if len(parts) > 1 else combined_response
            else:
                # Fallback: treat as refined response, extract analysis from context
                refined_response = combined_response
                analysis = "Response analyzed and improved based on identified gaps and errors."
            
            if verbose:
                print(f"🔍 Analysis {i} by {refiner.get_provider_name()}:")
                print(f"{analysis[:200]}...\n")
                print(f"🔧 Refinement {i} by {refiner.get_provider_name()}:")
                print(f"{refined_response[:300]}...\n")
            
            refinement_chain.append({
                "provider": refiner.get_provider_name(),
                "analysis": analysis,
                "response": refined_response,
                "stage": f"refinement_{i}"
            })
            
            current_response = refined_response
        
        if verbose:
            print("✅ Final Optimized Answer:\n")
            print(current_response)
        
        return {
            "question": prompt,
            "refinement_chain": refinement_chain,
            "final_answer": current_response
        }

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
