import asyncio
import os
from typing import List, Dict
from providers import GroqProvider, OllamaProvider, MistralProvider, BaseLLMProvider

# Timeout for API calls (seconds)
API_TIMEOUT = 60  # Increased to 60 seconds for slower APIs
OLLAMA_TIMEOUT = 120  # Longer timeout for local Ollama (can be slower)

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
        async def query_with_timeout(provider, prompt):
            try:
                return await asyncio.wait_for(provider.query(prompt), timeout=API_TIMEOUT)
            except asyncio.TimeoutError:
                return Exception(f"Timeout after {API_TIMEOUT}s")
            except Exception as e:
                return Exception(str(e))
        
        tasks = [
            query_with_timeout(provider, summary_prompt)
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
            try:
                critique = await asyncio.wait_for(
                    provider.query(critique_prompt),
                    timeout=API_TIMEOUT
                )
                critiques[provider_name] = critique
            except asyncio.TimeoutError:
                critiques[provider_name] = f"Timeout after {API_TIMEOUT}s"
            except Exception as e:
                critiques[provider_name] = f"Error: {str(e)}"

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
        try:
            final_answer = await asyncio.wait_for(
                synthesizer.query(synthesis_prompt),
                timeout=API_TIMEOUT
            )
            return final_answer
        except asyncio.TimeoutError:
            raise RuntimeError(f"Synthesis timed out after {API_TIMEOUT} seconds")
        except Exception as e:
            raise RuntimeError(f"Error during synthesis: {str(e)}")

    async def sequential_refine(self, prompt: str, verbose: bool = True) -> Dict:
        """Sequential refinement: LLM 1 generates, LLM 2+ refines by finding missing info and adding it."""
        self._current_prompt = prompt
        
        if len(self.providers) < 2:
            raise ValueError("Sequential refinement requires at least 2 providers")
        
        if verbose:
            print("🔄 Starting sequential refinement...\n")
        
        # Step 1: Get independent responses from both providers
        first_provider = self.providers[0]
        second_provider = self.providers[1]
        
        summary_prompt = f"""Provide a crisp, to-the-point answer. Be concise and direct - no fluff, just the essential information.

Question: {prompt}

Answer:"""
        
        # Get independent responses in parallel
        async def get_independent_response(provider):
            provider_name = provider.get_provider_name()
            # Use longer timeout for Ollama (local models can be slower)
            timeout = OLLAMA_TIMEOUT if provider_name == "Ollama" else API_TIMEOUT
            try:
                return await asyncio.wait_for(
                    provider.query(summary_prompt),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"{provider_name} timed out after {timeout} seconds. "
                    f"If using Ollama, make sure it's running and the model is downloaded: 'ollama pull {getattr(provider, 'default_model', 'llama3.1')}'"
                )
            except Exception as e:
                raise RuntimeError(f"Error from {provider_name}: {str(e)}")
        
        first_response, second_response = await asyncio.gather(
            get_independent_response(first_provider),
            get_independent_response(second_provider)
        )
        
        if verbose:
            print(f"📝 Independent response from {first_provider.get_provider_name()}:")
            print(f"{first_response[:300]}...\n")
            print(f"📝 Independent response from {second_provider.get_provider_name()}:")
            print(f"{second_response[:300]}...\n")
        
        # Step 2: Second provider analyzes first response and adds missing information
        refine_prompt = f"""Analyze the first response and find what's missing. Then create an optimized answer that combines both.

Question: {prompt}

First Response (from {first_provider.get_provider_name()}):
{first_response}

Second Response (from {second_provider.get_provider_name()}):
{second_response}

Task:
1. Identify what's missing in the first response that the second has
2. Find what's missing in the second response that the first has
3. Create ONE optimized answer that combines the best of both and fills all gaps

Format:
ANALYSIS:
[What's missing in each response]

OPTIMIZED RESPONSE:
[Combined answer with all information, no redundancy]"""
        
        refiner_name = second_provider.get_provider_name()
        # Use longer timeout for Ollama
        timeout = OLLAMA_TIMEOUT if refiner_name == "Ollama" else API_TIMEOUT
        try:
            combined_response = await asyncio.wait_for(
                second_provider.query(refine_prompt),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Refinement by {refiner_name} timed out after {API_TIMEOUT} seconds. "
                f"This might be due to slow API response or network issues."
            )
        except Exception as e:
            raise RuntimeError(
                f"Error during refinement by {refiner_name}: {str(e)}. "
                f"Check if the API key is valid and the service is available."
            )
        
        # Parse the response to extract analysis and optimized output
        if "ANALYSIS:" in combined_response and "OPTIMIZED RESPONSE:" in combined_response:
            parts = combined_response.split("OPTIMIZED RESPONSE:")
            analysis = parts[0].replace("ANALYSIS:", "").strip()
            optimized_response = parts[1].strip() if len(parts) > 1 else combined_response
        else:
            # Fallback: treat as optimized response
            optimized_response = combined_response
            analysis = "Response analyzed and optimized by combining information from both providers."
        
        if verbose:
            print(f"🔍 Analysis by {refiner_name}:")
            print(f"{analysis[:200]}...\n")
            print("✅ Final Optimized Answer:\n")
            print(optimized_response)
        
        return {
            "question": prompt,
            "independent_responses": {
                first_provider.get_provider_name(): first_response,
                second_provider.get_provider_name(): second_response
            },
            "analysis": analysis,
            "final_answer": optimized_response
        }
    
    async def summarize(self, prompt: str, verbose: bool = True) -> Dict:
        """Summarizer mode: Provides answers as short, easy-to-remember bullet points."""
        self._current_prompt = prompt
        
        if verbose:
            print("📋 Generating summary...\n")
        
        # Use first provider for summarization
        summarizer = self.providers[0]
        
        summary_prompt = f"""Provide a crisp answer in short, easy-to-remember bullet points.

Question: {prompt}

Format your answer as:
• Point 1 (concise)
• Point 2 (concise)
• Point 3 (concise)

Keep each point short (1-2 lines max). Make it easy to remember and understand.
Answer:"""
        
        try:
            summary = await asyncio.wait_for(
                summarizer.query(summary_prompt),
                timeout=API_TIMEOUT
            )
        except asyncio.TimeoutError:
            raise RuntimeError(f"Summary generation timed out after {API_TIMEOUT} seconds")
        except Exception as e:
            raise RuntimeError(f"Error generating summary: {str(e)}")
        
        if verbose:
            print("✅ Summary:\n")
            print(summary)
        
        return {
            "question": prompt,
            "summary": summary
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
