import asyncio
import sys
from dotenv import load_dotenv
from council import LLMCouncil

def main():
    """Main entry point for the LLM Council."""
    load_dotenv()
    
    if len(sys.argv) < 2:
        print("Usage: python main.py \"Your question here\"")
        sys.exit(1)
    
    prompt = " ".join(sys.argv[1:])
    
    council = LLMCouncil()
    
    try:
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

