import asyncio
import sys
from dotenv import load_dotenv
from council import LLMCouncil

def main():
    """Main entry point for the LLM Council."""
    load_dotenv()
    
    if len(sys.argv) < 2:
        print("Usage: python main.py \"Your question here\" [mode]")
        print("\nModes:")
        print("  (default) - All: All LLMs respond, cross-critique, and synthesize")
        print("  sequential - Sequential Refinement: Both LLMs respond independently, then combine")
        print("  summarize - Summarizer: Short bullet points")
        sys.exit(1)
    
    # Check if last argument is a mode
    if len(sys.argv) > 2 and sys.argv[-1] in ["sequential", "summarize"]:
        mode = sys.argv[-1]
        prompt = " ".join(sys.argv[1:-1])
    else:
        mode = "all"
        prompt = " ".join(sys.argv[1:])
    
    council = LLMCouncil()
    
    try:
        if mode == "sequential":
            result = asyncio.run(council.sequential_refine(prompt))
            print("\n📝 Independent Responses:")
            for provider, response in result["independent_responses"].items():
                print(f"\n{provider}:")
                print(response)
            print("\n🔍 Analysis:")
            print(result.get("analysis", ""))
            print("\n✅ Final Optimized Answer:")
            print(result["final_answer"])
        elif mode == "summarize":
            result = asyncio.run(council.summarize(prompt))
            print("\n📋 Summary:")
            print(result["summary"])
        else:
            result = asyncio.run(council.consult(prompt))
        return result
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

