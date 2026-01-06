import streamlit as st
import asyncio
import sys
import warnings
import logging
import html
from dotenv import load_dotenv
from council import LLMCouncil

# Suppress harmless Streamlit websocket warnings (these are internal Streamlit errors, not app errors)
warnings.filterwarnings("ignore", category=UserWarning)
# Reduce noise from tornado websocket errors (common when browser closes/refreshes)
logging.getLogger("tornado.access").setLevel(logging.WARNING)
logging.getLogger("tornado.application").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

# Page config - light theme
st.set_page_config(
    page_title="LLM Council Chat",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Enhanced light theme CSS
st.markdown("""
    <style>
    /* Main background */
    .main {
        background-color: #ffffff;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Assistant message background - light for black text */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAssistant"]) {
        background-color: #f8f9fa;
        border-radius: 8px;
    }
    
    /* Assistant message text - black */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAssistant"]) .stMarkdown {
        color: #212529 !important;
    }
    
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAssistant"]) .stMarkdown p {
        color: #212529 !important;
    }
    
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAssistant"]) .stMarkdown h1,
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAssistant"]) .stMarkdown h2,
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAssistant"]) .stMarkdown h3 {
        color: #212529 !important;
    }
    
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAssistant"]) .stMarkdown strong {
        color: #212529 !important;
    }
    
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAssistant"]) .stMarkdown li {
        color: #212529 !important;
    }
    
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAssistant"]) .stCaption {
        color: #6c757d !important;
    }
    
    /* User message styling */
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        background-color: #e7f3ff;
        padding: 0.75rem;
        border-radius: 8px;
    }
    
    /* Input field */
    .stTextInput > div > div > input {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        color: #212529;
    }
    
    /* Selectbox */
    .stSelectbox > div > div > select {
        background-color: #ffffff;
        color: #212529;
        border: 1px solid #dee2e6;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #212529;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #ffffff;
        color: #212529;
    }
    
    /* Progress bar text */
    .stProgress > div > div > div {
        color: #212529;
    }
    
    /* Button */
    .stButton > button {
        background-color: #0d6efd;
        color: #ffffff;
        border: none;
    }
    
    .stButton > button:hover {
        background-color: #0b5ed7;
    }
    
    /* Provider Response Cards */
    .provider-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-left: 4px solid #0d6efd;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .provider-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    
    .provider-card.groq {
        border-left-color: #10b981;
        background: linear-gradient(135deg, #ecfdf5 0%, #ffffff 100%);
    }
    
    .provider-card.mistral {
        border-left-color: #8b5cf6;
        background: linear-gradient(135deg, #f5f3ff 0%, #ffffff 100%);
    }
    
    .provider-card.ollama {
        border-left-color: #f59e0b;
        background: linear-gradient(135deg, #fffbeb 0%, #ffffff 100%);
    }
    
    .provider-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .provider-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 600;
        font-size: 1.1rem;
        color: #1f2937;
    }
    
    .provider-icon {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    .provider-icon.groq {
        background: #10b981;
        color: white;
    }
    
    .provider-icon.mistral {
        background: #8b5cf6;
        color: white;
    }
    
    .provider-icon.ollama {
        background: #f59e0b;
        color: white;
    }
    
    .provider-content {
        color: #374151;
        line-height: 1.7;
        font-size: 0.95rem;
    }
    
    /* Final Answer Highlight */
    .final-answer {
        background: linear-gradient(135deg, #dbeafe 0%, #eff6ff 100%);
        border: 2px solid #3b82f6;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
    }
    
    .final-answer-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
        color: #1e40af;
        font-weight: 600;
        font-size: 1.2rem;
    }
    
    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
        color: #1f2937;
        font-weight: 600;
        font-size: 1.3rem;
    }
    
    /* Divider */
    .response-divider {
        height: 1px;
        background: linear-gradient(to right, transparent, #e5e7eb, transparent);
        margin: 1.5rem 0;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "council" not in st.session_state:
    with st.spinner("Initializing LLM Council..."):
        try:
            st.session_state.council = LLMCouncil()
            # Get available provider names
            st.session_state.available_providers = [
                provider.get_provider_name() for provider in st.session_state.council.providers
            ]
        except Exception as e:
            st.error(f"Failed to initialize LLM Council: {e}")
            st.stop()

# Header
st.title("🤖 LLM Council Chat")
st.markdown("Ask questions and get crisp, to-the-point answers from multiple LLMs")

# Mode selection dropdown - All or individual providers
mode_options = ["Sequential Refinement", "Summarizer", "All"] + st.session_state.available_providers
mode = st.selectbox(
    "Response Mode",
    mode_options,
    help="Sequential Refinement: Both LLMs respond independently, then combine. Summarizer: Short bullet points. All: All LLMs synthesize. Provider: Single provider only."
)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            stored_mode = message.get("mode", "All")
            
            # Show based on current mode selection
            if mode == "Sequential Refinement" and stored_mode == "Sequential Refinement":
                if "independent_responses" in message:
                    # New format: Show independent responses with styled cards
                    st.markdown('<div class="section-header">📋 Independent Responses</div>', unsafe_allow_html=True)
                    
                    provider_icons = {
                        "Groq": "🚀",
                        "Mistral": "💜",
                        "Ollama": "🦙"
                    }
                    
                    for provider, response in message["independent_responses"].items():
                        icon = provider_icons.get(provider, "🤖")
                        provider_lower = provider.lower()
                        
                        card_header = f"""
                        <div class="provider-card {provider_lower}">
                            <div class="provider-header">
                                <div class="provider-icon {provider_lower}">{icon}</div>
                                <div class="provider-badge">{provider}</div>
                            </div>
                        """
                        st.markdown(card_header, unsafe_allow_html=True)
                        st.markdown('<div class="provider-content">', unsafe_allow_html=True)
                        st.markdown(response)
                        st.markdown('</div></div>', unsafe_allow_html=True)
                    
                    # Show analysis
                    if "analysis" in message:
                        with st.expander("🔍 Analysis - What's Missing"):
                            st.markdown(message["analysis"])
                    
                    # Show optimized answer
                    final_answer_header = """
                    <div class="final-answer">
                        <div class="final-answer-header">
                            <span>✨</span>
                            <span>Final Optimized Answer</span>
                        </div>
                    """
                    st.markdown(final_answer_header, unsafe_allow_html=True)
                    st.markdown('<div class="provider-content">', unsafe_allow_html=True)
                    st.markdown(message["content"])
                    st.markdown('</div></div>', unsafe_allow_html=True)
                elif "refinement_chain" in message:
                    # Old format: Show refinement chain (backward compatibility)
                    st.markdown("### Refinement Chain:")
                    for step in message["refinement_chain"]:
                        if step["stage"] == "initial":
                            st.markdown(f"**Initial Response ({step['provider']}):**")
                            st.markdown(step["response"])
                        else:
                            refinement_num = step['stage'].split('_')[1]
                            st.markdown(f"**Refinement {refinement_num} ({step['provider']}):**")
                            
                            # Show analysis in an expander
                            with st.expander(f"🔍 Analysis - What's Missing/Wrong"):
                                st.markdown(step["analysis"])
                            
                            # Show refined response
                            st.markdown("**Refined Output:**")
                            st.markdown(step["response"])
                        st.markdown("---")
                    
                    st.markdown("### ✅ Final Optimized Answer:")
                    st.markdown(message["content"])
            
            elif mode == "Summarizer" and stored_mode == "Summarizer":
                # Show summary as bullet points
                if "summary" in message:
                    st.markdown(message["summary"])
                else:
                    st.markdown(message["content"])
            
            elif mode == "All" and stored_mode == "All" and "details" in message:
                # Show all mode: individual responses first, then optimized
                st.markdown('<div class="section-header">📋 Individual Responses</div>', unsafe_allow_html=True)
                
                provider_icons = {
                    "Groq": "🚀",
                    "Mistral": "💜",
                    "Ollama": "🦙"
                }
                
                for provider, response in message["details"]["responses"].items():
                    icon = provider_icons.get(provider, "🤖")
                    provider_lower = provider.lower()
                    
                    card_header = f"""
                    <div class="provider-card {provider_lower}">
                        <div class="provider-header">
                            <div class="provider-icon {provider_lower}">{icon}</div>
                            <div class="provider-badge">{provider}</div>
                        </div>
                    """
                    st.markdown(card_header, unsafe_allow_html=True)
                    st.markdown('<div class="provider-content">', unsafe_allow_html=True)
                    st.markdown(response)
                    st.markdown('</div></div>', unsafe_allow_html=True)
                
                # Show critiques if available
                if "critiques" in message["details"]:
                    st.markdown('<div class="section-header">🔍 Critiques</div>', unsafe_allow_html=True)
                    for provider, critique in message["details"]["critiques"].items():
                        with st.expander(f"🔍 {provider} Critique"):
                            st.markdown(critique)
                
                # Final answer
                final_answer_header = """
                <div class="final-answer">
                    <div class="final-answer-header">
                        <span>✨</span>
                        <span>Final Optimized Answer</span>
                    </div>
                """
                st.markdown(final_answer_header, unsafe_allow_html=True)
                st.markdown('<div class="provider-content">', unsafe_allow_html=True)
                st.markdown(message["content"])
                st.markdown('</div></div>', unsafe_allow_html=True)
            
            elif mode != "All" and message.get("provider") == mode:
                # Show specific provider's independent response
                st.markdown(message["content"])
            
            elif mode != "All" and stored_mode == "All" and "details" in message and mode in message["details"]["responses"]:
                # Show this provider's response from stored council data
                st.markdown(message["details"]["responses"][mode])
            
            else:
                # Fallback: show the content
                st.markdown(message["content"])
        else:
            # User messages always show
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your question here..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response based on mode
    with st.chat_message("assistant"):
        try:
            if mode == "Sequential Refinement":
                # Show progress immediately
                status_placeholder = st.empty()
                status_placeholder.info("🔄 Processing... This may take 30-60 seconds.")
                
                # Run sequential refinement (non-blocking display)
                try:
                    result = asyncio.run(st.session_state.council.sequential_refine(prompt, verbose=False))
                    status_placeholder.empty()
                except Exception as e:
                    status_placeholder.empty()
                    raise
                
                # Show independent responses with styled cards
                st.markdown('<div class="section-header">📋 Independent Responses</div>', unsafe_allow_html=True)
                
                provider_icons = {
                    "Groq": "🚀",
                    "Mistral": "💜",
                    "Ollama": "🦙"
                }
                
                for provider, response in result["independent_responses"].items():
                    icon = provider_icons.get(provider, "🤖")
                    provider_lower = provider.lower()
                    
                    # Card header with HTML
                    card_header = f"""
                    <div class="provider-card {provider_lower}">
                        <div class="provider-header">
                            <div class="provider-icon {provider_lower}">{icon}</div>
                            <div class="provider-badge">{provider}</div>
                        </div>
                    """
                    st.markdown(card_header, unsafe_allow_html=True)
                    
                    # Content with markdown support
                    st.markdown(f'<div class="provider-content">', unsafe_allow_html=True)
                    st.markdown(response)
                    st.markdown('</div></div>', unsafe_allow_html=True)
                
                # Show analysis
                if "analysis" in result:
                    with st.expander("🔍 Analysis - What's Missing"):
                        st.markdown(result["analysis"])
                
                # Show optimized answer with highlight
                final_answer_header = """
                <div class="final-answer">
                    <div class="final-answer-header">
                        <span>✨</span>
                        <span>Final Optimized Answer</span>
                    </div>
                """
                st.markdown(final_answer_header, unsafe_allow_html=True)
                st.markdown('<div class="provider-content">', unsafe_allow_html=True)
                st.markdown(result["final_answer"])
                st.markdown('</div></div>', unsafe_allow_html=True)
                
                assistant_message = {
                    "role": "assistant",
                    "content": result["final_answer"],
                    "mode": "Sequential Refinement",
                    "provider": None,
                    "independent_responses": result["independent_responses"],
                    "analysis": result.get("analysis", "")
                }
            
            elif mode == "Summarizer":
                status_placeholder = st.empty()
                status_placeholder.info("🔄 Generating summary...")
                try:
                    # Run summarizer
                    result = asyncio.run(st.session_state.council.summarize(prompt, verbose=False))
                    status_placeholder.empty()
                except Exception as e:
                    status_placeholder.empty()
                    raise
                
                # Show summary as bullet points
                st.markdown(result["summary"])
                
                assistant_message = {
                    "role": "assistant",
                    "content": result["summary"],
                    "mode": "Summarizer",
                    "provider": None,
                    "summary": result["summary"]
                }
                
            elif mode == "All":
                status_placeholder = st.empty()
                status_placeholder.info("🔄 Consulting all LLMs... This may take 30-60 seconds.")
                try:
                    # Run full council consultation
                    result = asyncio.run(st.session_state.council.consult(prompt, verbose=False))
                    status_placeholder.empty()
                except Exception as e:
                    status_placeholder.empty()
                    raise
                
                # Show all individual responses with styled cards
                st.markdown('<div class="section-header">📋 Individual Responses</div>', unsafe_allow_html=True)
                
                provider_icons = {
                    "Groq": "🚀",
                    "Mistral": "💜",
                    "Ollama": "🦙"
                }
                
                for provider, response in result["responses"].items():
                    icon = provider_icons.get(provider, "🤖")
                    provider_lower = provider.lower()
                    
                    card_header = f"""
                    <div class="provider-card {provider_lower}">
                        <div class="provider-header">
                            <div class="provider-icon {provider_lower}">{icon}</div>
                            <div class="provider-badge">{provider}</div>
                        </div>
                    """
                    st.markdown(card_header, unsafe_allow_html=True)
                    st.markdown('<div class="provider-content">', unsafe_allow_html=True)
                    st.markdown(response)
                    st.markdown('</div></div>', unsafe_allow_html=True)
                
                # Show critiques
                st.markdown("### Critiques:")
                for provider, critique in result["critiques"].items():
                    with st.expander(f"🔍 {provider} Critique"):
                        st.markdown(critique)
                
                st.markdown("---")
                
                # Then show optimized answer with highlight
                final_answer_header = """
                <div class="final-answer">
                    <div class="final-answer-header">
                        <span>✨</span>
                        <span>Final Optimized Answer</span>
                    </div>
                """
                st.markdown(final_answer_header, unsafe_allow_html=True)
                st.markdown('<div class="provider-content">', unsafe_allow_html=True)
                st.markdown(result["final_answer"])
                st.markdown('</div></div>', unsafe_allow_html=True)
                
                assistant_message = {
                    "role": "assistant",
                    "content": result["final_answer"],
                    "mode": "All",
                    "provider": None,
                    "details": {
                        "responses": result["responses"],
                        "critiques": result["critiques"]
                    }
                }
            else:
                status_placeholder = st.empty()
                status_placeholder.info(f"🔄 Querying {mode}...")
                try:
                    # Query only the selected provider (independent mode)
                    selected_provider = None
                    for provider in st.session_state.council.providers:
                        if provider.get_provider_name() == mode:
                            selected_provider = provider
                            break
                    
                    if selected_provider:
                        response = asyncio.run(selected_provider.query(prompt))
                        status_placeholder.empty()
                        st.markdown(response)
                        
                        assistant_message = {
                            "role": "assistant",
                            "content": response,
                            "mode": "Explicit",
                            "provider": mode
                        }
                    else:
                        raise ValueError(f"Provider {mode} not found")
                except Exception as e:
                    status_placeholder.empty()
                    raise
            
            st.session_state.messages.append(assistant_message)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.error(error_msg)
            import traceback
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
            
            # Show error in chat
            st.markdown(f"**❌ Error:** {error_msg}")
            
            assistant_message = {
                "role": "assistant",
                "content": error_msg,
                "mode": mode,
                "provider": mode if mode != "All" else None,
                "error": True
            }
            st.session_state.messages.append(assistant_message)

# Sidebar info
with st.sidebar:
    st.header("About")
    st.markdown("""
    **LLM Council** uses multiple AI models as summarizers to generate crisp, to-the-point answers.
    
    **What is LLM?**
    **LLM = Large Language Model**
    
    AI systems trained on vast amounts of text that can:
    - Understand and generate human-like text
    - Answer questions
    - Summarize information
    - Examples: ChatGPT, Claude, Llama, Mistral
    
    **Models Used:**
    - **Groq**: Uses Llama 3.1 70B model (fast cloud inference)
    - **Mistral**: Mistral AI's models (if Ollama disabled)
    - **Ollama**: Runs Llama locally on your computer (if enabled)
    
    **Response Modes:**
    - **Sequential Refinement**: Both LLMs respond independently → Second LLM finds missing info and combines into optimized answer
    - **Summarizer**: Short, easy-to-remember bullet points
    - **All**: All LLMs respond independently, then synthesize into one crisp answer
    - **Provider Name**: Single provider only
    
    **Ollama Troubleshooting:**
    If Ollama times out:
    1. Make sure Ollama is running: `ollama serve`
    2. Download the model: `ollama pull llama3.1`
    3. Check if Ollama is accessible at http://localhost:11434
    """)
    
    st.markdown(f"**Available Providers:** {', '.join(st.session_state.available_providers)}")
    
    # Show model details if available
    try:
        if st.session_state.council.providers:
            st.markdown("**Active Models:**")
            for provider in st.session_state.council.providers:
                provider_name = provider.get_provider_name()
                if hasattr(provider, 'default_model'):
                    st.markdown(f"- {provider_name}: {provider.default_model}")
                else:
                    st.markdown(f"- {provider_name}")
    except:
        pass
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("Clear Memory"):
            if "council" in st.session_state:
                st.session_state.council.memory.clear()
            st.success("Conversation memory cleared!")
    
    st.markdown("---")
    st.markdown("**Features:**")
    st.markdown("✅ Response caching (saves API costs)")
    st.markdown("✅ Conversation memory (context-aware)")
    
    st.markdown("---")
    st.markdown("**Performance Tips:**")
    st.markdown("""
    - **First load**: May take 5-10s (initialization)
    - **Cached queries**: Instant (0.1s)
    - **New queries**: 10-60s (depends on mode)
    - **Ollama**: Can be slow (120s timeout)
    - **Tip**: Use Groq only for fastest responses
    """)
