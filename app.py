import streamlit as st
from langchain_setup import agent_executor

st.set_page_config(page_title="LangChain Assistant", layout="wide")

# Sidebar config
with st.sidebar:
    st.title("🧠 LangChain Agent")
    st.markdown("Built with **OpenAI + Tavily + NLLB Translation**")

    st.markdown("## 🛠️ Tools")
    st.write("✅ Web Search\n ✅ Summarization\n ✅ Translation")

# Main chat area
st.title("🗣️ AI Task Assistant")

try:
    from langchain_setup import agent_executor
except Exception as e:
    st.error(f"🔥 App failed to load LangChain agent: {e}")
    st.stop()

# If it loads, you're good to go
st.success("✅ App initialized successfully.")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Type a task...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "text": user_input})

    with st.spinner("Thinking..."):
        result = agent_executor.invoke({"input": user_input})

    st.session_state.chat_history.append({"role": "ai", "text": result.get("output", "⚠️ No response")})

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.chat_message("user").write(message["text"])
    else:
        st.chat_message("assistant").write(message["text"])
# Footer
st.markdown("---")