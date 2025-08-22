import streamlit as st
import asyncio
from typing import List, Dict
import json
from datetime import datetime

# Import AI Libraries
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage

# Page Config
st.set_page_config(
    page_title="Multi-AI Chat Hub",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {}

# Sidebar for API Keys
with st.sidebar:
    st.title("🔑 API Configuration")
    
    with st.expander("OpenAI (ChatGPT)", expanded=False):
        openai_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
        if openai_key:
            st.session_state.api_keys['openai'] = openai_key
    
    with st.expander("Google (Gemini)", expanded=False):
        google_key = st.text_input("Google API Key", type="password", key="google_key")
        if google_key:
            st.session_state.api_keys['google'] = google_key
    
    with st.expander("Anthropic (Claude)", expanded=False):
        anthropic_key = st.text_input("Anthropic API Key", type="password", key="anthropic_key")
        if anthropic_key:
            st.session_state.api_keys['anthropic'] = anthropic_key
    
    st.divider()
    
    # Model Selection
    st.title("🎯 Select AI Models")
    selected_models = st.multiselect(
        "Choose models to query:",
        ["GPT-4o", "GPT-4o-mini", "Gemini-1.5-Pro", "Gemini-1.5-Flash", "Claude-3.5-Sonnet"],
        default=["GPT-4o-mini", "Gemini-1.5-Flash"]
    )
    
    # Advanced Settings
    with st.expander("⚙️ Advanced Settings"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
        max_tokens = st.slider("Max Tokens", 100, 4000, 1000)
        
    # Clear Chat Button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main Chat Interface
st.title("🤖 Multi-AI Chat Hub")
st.caption("Query multiple AI models simultaneously")

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.write(f"**{message['model']}**")
            st.write(message["content"])

# Initialize AI models
def get_llm(model_name: str):
    """Initialize LLM based on model name"""
    try:
        if "GPT" in model_name:
            if 'openai' not in st.session_state.api_keys:
                return None
            model = "gpt-4o" if "4o" in model_name and "mini" not in model_name else "gpt-4o-mini"
            return ChatOpenAI(
                model=model,
                api_key=st.session_state.api_keys['openai'],
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        elif "Gemini" in model_name:
            if 'google' not in st.session_state.api_keys:
                return None
            model = "gemini-1.5-pro" if "Pro" in model_name else "gemini-1.5-flash"
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=st.session_state.api_keys['google'],
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        
        elif "Claude" in model_name:
            if 'anthropic' not in st.session_state.api_keys:
                return None
            return ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                api_key=st.session_state.api_keys['anthropic'],
                temperature=temperature,
                max_tokens=max_tokens
            )
    except Exception as e:
        st.error(f"Error initializing {model_name}: {str(e)}")
        return None

async def query_model_async(model_name: str, prompt: str):
    """Query a single model asynchronously"""
    try:
        llm = get_llm(model_name)
        if llm is None:
            return {"model": model_name, "response": "❌ API key not configured", "error": True}
        
        # Convert chat history to messages
        messages = []
        for msg in st.session_state.messages[-10:]:  # Last 10 messages for context
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg.get("model") == model_name:  # Only include responses from same model
                messages.append(AIMessage(content=msg["content"]))
        
        # Add current prompt
        messages.append(HumanMessage(content=prompt))
        
        # Get response
        response = await asyncio.to_thread(llm.invoke, messages)
        return {"model": model_name, "response": response.content, "error": False}
    
    except Exception as e:
        return {"model": model_name, "response": f"❌ Error: {str(e)}", "error": True}

async def query_all_models(prompt: str, models: List[str]):
    """Query all selected models concurrently"""
    tasks = [query_model_async(model, prompt) for model in models]
    responses = await asyncio.gather(*tasks)
    return responses

# Chat Input
if prompt := st.chat_input("Ask your question to multiple AIs..."):
    # Check if models are selected
    if not selected_models:
        st.error("Please select at least one AI model from the sidebar!")
    else:
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Add to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Create placeholder for responses
        response_container = st.container()
        
        with response_container:
            # Show loading spinner
            with st.spinner(f"Querying {len(selected_models)} models..."):
                # Query all models
                responses = asyncio.run(query_all_models(prompt, selected_models))
                
                # Display responses in columns
                cols = st.columns(len(responses))
                
                for col, response in zip(cols, responses):
                    with col:
                        with st.chat_message("assistant", avatar="🤖"):
                            st.write(f"**{response['model']}**")
                            if response['error']:
                                st.error(response['response'])
                            else:
                                st.write(response['response'])
                                # Add to history
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "model": response['model'],
                                    "content": response['response']
                                })

# Footer with instructions
with st.expander("📖 How to Use", expanded=False):
    st.markdown("""
    ### Quick Start:
    1. **Add API Keys** in the sidebar (at least one)
    2. **Select Models** you want to query
    3. **Type your question** and press Enter
    4. **Compare responses** side by side
    
    ### Features:
    - ✅ Query multiple AI models simultaneously
    - ✅ Maintains conversation context
    - ✅ Side-by-side comparison
    - ✅ Adjustable parameters (temperature, max tokens)
    - ✅ Chat history preservation
    
    ### Getting API Keys:
    - **OpenAI**: https://platform.openai.com/api-keys
    - **Google**: https://makersuite.google.com/app/apikey
    - **Anthropic**: https://console.anthropic.com/settings/keys
    
    ### Notes:
    - For ChatGPT Plus users: You still need an API key (separate from Plus subscription)
    - DeepSeek & Qwen: Can be added easily - just need their API endpoints
    """)

# Status Bar
st.sidebar.divider()
st.sidebar.caption(f"💡 Connected Models: {len([k for k in st.session_state.api_keys if st.session_state.api_keys[k]])}")
st.sidebar.caption(f"💬 Messages in History: {len(st.session_state.messages)}")