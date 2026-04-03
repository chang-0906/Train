import os
import streamlit as st
from langfuse.openai import openai

# 先在環境變數裡設定：
# LANGFUSE_PUBLIC_KEY=pk-lf-...
# LANGFUSE_SECRET_KEY=sk-lf-...
# LANGFUSE_BASE_URL=https://cloud.langfuse.com
# 如果你用美國區，則改成 https://us.cloud.langfuse.com

st.set_page_config(page_title="My Modal LLM Chat", page_icon="💬")
st.title("My Modal LLM Chat")

# 這裡依然是你的 Modal / Ollama OpenAI-compatible endpoint
client = openai.OpenAI(
    base_url="https://chang-0906--ollama-gguf-hf-import-ollamaggufserver-api.modal.run/v1",
    api_key="ollama",
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "請一律用繁體中文回答。"}
    ]

if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit-demo-session"

if "user_id" not in st.session_state:
    st.session_state.user_id = "demo-user"

for msg in st.session_state.messages[1:]:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

prompt = st.chat_input("輸入訊息...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    resp = client.chat.completions.create(
        name="modal-ollama-chat",
        model="llama31-abliterated-q4km:latest",
        messages=st.session_state.messages,
        metadata={
            "app": "streamlit-modal-chat",
            "provider": "modal-ollama",
        },
        extra_body={
            # 這些不是 OpenAI 官方欄位，而是 Langfuse wrapper 會讀的追蹤資訊
            "langfuse_user_id": st.session_state.user_id,
            "langfuse_session_id": st.session_state.session_id,
            "langfuse_tags": ["streamlit", "modal", "ollama"],
        },
    )

    reply = resp.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": reply})

    with st.chat_message("assistant"):
        st.markdown(reply)