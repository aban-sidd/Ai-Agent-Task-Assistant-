import streamlit as st
from langchain_setup import build_agent_executor

st.set_page_config(page_title="LangChain Assistant", layout="wide")

# Initialize session state
if "connected" not in st.session_state:
    st.session_state.connected = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "OpenAI"

# Sidebar config
with st.sidebar:
    st.title("🧠 LangChain Agent")
    st.markdown("Built with **OpenAI + Tavily + NLLB Translation**")

    st.markdown("## 🛠️ Tools")
    st.write("✅ Web Search\n ✅ Summarization\n ✅ Translation")

      # Select model
    model_choice = st.selectbox("Select Model", ["OpenAI", "Groq"])

    # Reset chat if model is changed
    if model_choice != st.session_state.selected_model:
        st.session_state.selected_model = model_choice
        st.session_state.connected = False
        st.session_state.chat_history = []

    # API key input
    api_key = st.text_input(
        f"{model_choice} API Key",
        type="password",
        placeholder="Paste your API key here"
    )

   # Connect button
    if st.button("🔌 Connect"):
        if not api_key:
            st.warning("Please enter a valid API key.")
        else:
            try:
                st.session_state.agent_executor = build_agent_executor(model_choice, api_key)
                st.session_state.connected = True
                st.success(f"✅ Connected to {model_choice}")
            except Exception as e:
                st.session_state.connected = False
                st.error(f"❌ Error initializing agent: {e}")

# Stop if not connected
if not st.session_state.connected:
    st.info("🔐 Please connect using the sidebar to start chatting.")
    st.stop()

    
# Main chat area
st.title("🗣️ AI Task Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Type a task...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "text": user_input})

    with st.spinner("Thinking..."):
        result = st.session_state.agent_executor.invoke({"input": user_input})

    st.session_state.chat_history.append({"role": "ai", "text": result.get("output", "⚠️ No response")})

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.chat_message("user").write(message["text"])
    else:
        st.chat_message("assistant").write(message["text"])
# Footer
st.markdown("---")