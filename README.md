# LLM Council

A multi-LLM system that queries Groq, Ollama, and Mistral independently, then cross-verifies responses to provide the most optimal answer.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Get API keys:

   **Mistral API Key:**
   - Go to https://console.mistral.ai/
   - Sign up or log in
   - Navigate to "API Keys" section
   - Click "Create API Key"
   - Copy the generated key

   **Groq API Key:**
   - Go to https://console.groq.com/
   - Sign up or log in
   - Navigate to "API Keys" section
   - Create a new API key
   - Copy the generated key

   **Ollama:**
   - Install Ollama locally: https://ollama.ai/
   - Run `ollama pull llama3.1` (or your preferred model)
   - No API key needed, runs locally

3. Create a `.env` file in the project root:

   **Cost-effective setup (using Ollama instead of Mistral API):**
   ```
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=llama-3.1-8b-instant
   USE_OLLAMA=true
   OLLAMA_MODEL=mistral
   OLLAMA_BASE_URL=http://localhost:11434
   ```

   **Full setup (with Mistral API):**
   ```
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=llama-3.1-8b-instant
   MISTRAL_API_KEY=your_mistral_api_key_here
   USE_OLLAMA=false
   ```

   **Configuration Options:**
   - `GROQ_MODEL`: Model to use for Groq (default: `llama-3.1-70b-versatile`)
   - `USE_OLLAMA`: Set to `true` to use Ollama instead of Mistral API (saves costs)
   - `OLLAMA_MODEL`: Model to use for Ollama (default: `llama3.1`)
   - `OLLAMA_BASE_URL`: Ollama server URL (default: `http://localhost:11434`)

4. Run the council:

   **Streamlit Web Interface (Recommended):**
   ```bash
   streamlit run app.py
   ```
   Then open your browser to the URL shown (usually http://localhost:8501)

   **Command Line Interface:**
   ```bash
   python main.py "Your question here"
   ```

## How It Works

1. **Independent Queries**: Each LLM (Groq, Ollama, Mistral) independently answers the question
2. **Cross-Verification**: Responses are compared and verified against each other
3. **Optimal Answer**: The system selects the most consistent and well-verified answer

