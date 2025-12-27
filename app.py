import streamlit as st
import asyncio
from dotenv import load_dotenv
from council import LLMCouncil

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
    
    /* Assistant message background - dark for white text */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAssistant"]) {
        background-color: #1a1a1a;
        border-radius: 8px;
    }
    
    /* Assistant message text - white */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAssistant"]) .stMarkdown {
        color: #ffffff !important;
    }
    
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAssistant"]) .stMarkdown p {
        color: #ffffff !important;
    }
    
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAssistant"]) .stMarkdown h1,
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAssistant"]) .stMarkdown h2,
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAssistant"]) .stMarkdown h3 {
        color: #ffffff !important;
    }
    
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAssistant"]) .stMarkdown strong {
        color: #ffffff !important;
    }
    
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAssistant"]) .stMarkdown li {
        color: #ffffff !important;
    }
    
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAssistant"]) .stCaption {
        color: #cccccc !important;
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
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "council" not in st.session_state:
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
st.markdown("Ask questions and get verified answers from multiple LLMs")

# Mode selection dropdown - All or individual providers
mode_options = ["All"] + st.session_state.available_providers
mode = st.selectbox(
    "Response Mode",
    mode_options,
    help="All: Show all individual responses then optimized answer. Select a provider for that provider's answer only."
)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            stored_mode = message.get("mode", "All")
            
            # Show based on current mode selection
            if mode == "All" and stored_mode == "All" and "details" in message:
                # Show all mode: individual responses first, then optimized
                st.markdown("### Individual Responses:")
                for provider, response in message["details"]["responses"].items():
                    st.markdown(f"**{provider}:**")
                    st.markdown(response)
                    st.markdown("---")
                
                # Show confidence scores
                st.markdown("### Confidence Scores:")
                for provider, score in sorted(
                    message["details"]["confidence_scores"].items(),
                    key=lambda x: x[1],
                    reverse=True
                ):
                    st.progress(score, text=f"{provider}: {score:.1%}")
                
                st.markdown("---")
                st.markdown("### ✅ Optimized Answer:")
                st.markdown(message["content"])
                st.caption(f"Selected Provider: {message['details']['selected_provider']} | Confidence: {message['details']['confidence']:.1%}")
            
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
        with st.spinner("Processing your question..."):
            try:
                if mode == "All":
                    # Run full council consultation
                    result = asyncio.run(st.session_state.council.consult(prompt, verbose=False))
                    
                    # Show all individual responses first
                    st.markdown("### Individual Responses:")
                    for provider, response in result["responses"].items():
                        st.markdown(f"**{provider}:**")
                        st.markdown(response)
                        st.markdown("---")
                    
                    # Show confidence scores
                    st.markdown("### Confidence Scores:")
                    for provider, score in sorted(
                        result["confidence_scores"].items(),
                        key=lambda x: x[1],
                        reverse=True
                    ):
                        st.progress(score, text=f"{provider}: {score:.1%}")
                    
                    st.markdown("---")
                    
                    # Then show optimized answer
                    st.markdown("### ✅ Optimized Answer:")
                    st.markdown(result["optimal_answer"])
                    st.caption(f"Selected Provider: {result['selected_provider']} | Confidence: {result['confidence']:.1%}")
                    
                    assistant_message = {
                        "role": "assistant",
                        "content": result["optimal_answer"],
                        "mode": "All",
                        "provider": None,
                        "details": {
                            "responses": result["responses"],
                            "confidence_scores": result["confidence_scores"],
                            "selected_provider": result["selected_provider"],
                            "confidence": result["confidence"]
                        }
                    }
                else:
                    # Query only the selected provider (independent mode)
                    selected_provider = None
                    for provider in st.session_state.council.providers:
                        if provider.get_provider_name() == mode:
                            selected_provider = provider
                            break
                    
                    if selected_provider:
                        response = asyncio.run(selected_provider.query(prompt))
                        st.markdown(response)
                        
                        assistant_message = {
                            "role": "assistant",
                            "content": response,
                            "mode": "Explicit",
                            "provider": mode
                        }
                    else:
                        raise ValueError(f"Provider {mode} not found")
                
                st.session_state.messages.append(assistant_message)
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "mode": mode,
                    "provider": mode if mode != "All" else None
                })

# Sidebar info
with st.sidebar:
    st.header("About")
    st.markdown("""
    **LLM Council** uses multiple AI models to:
    1. Independently answer your question
    2. Cross-verify responses
    3. Select the most optimal answer
    
    **Response Modes:**
    - **All**: Shows all individual responses, then the optimized answer
    - **Provider Name** (Groq/Ollama/Mistral): Shows only that provider's answer
    """)
    
    st.markdown(f"**Available Providers:** {', '.join(st.session_state.available_providers)}")
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
