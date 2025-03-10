
import streamlit as st
import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import json

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""

# Sidebar - User API Key & Model Settings
st.sidebar.header("🔑 OpenAI API Key")
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
if api_key:
    st.session_state.openai_api_key = api_key

st.sidebar.header("⚙️ Model Settings")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 50, 500, 200)

# Initialize OpenAI Chat Model
if st.session_state.openai_api_key:
    chat_model = ChatOpenAI(temperature=temperature, model_name="gpt-4", openai_api_key=st.session_state.openai_api_key)
    memory = ConversationBufferMemory()
    chatbot = ConversationalRetrievalChain(llm=chat_model, memory=memory)

    st.title("🤖 Chatbot")
    st.write("Welcome! Start chatting with the bot below.")

    # Chat Interface
    user_input = st.text_input("You:", "")
    if st.button("Send") and user_input:
        with st.spinner("Thinking..."):
            response = chatbot.run(user_input)
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Bot", response))
            st.write(f"**Bot:** {response}")

    # Display Chat History
    st.subheader("📜 Chat History")
    for user, message in st.session_state.chat_history:
        st.write(f"**{user}:** {message}")

    # Download Chat History
    st.sidebar.subheader("⬇️ Download Chat")
    export_format = st.sidebar.selectbox("Format", ["JSON", "CSV", "PDF"])
    if st.sidebar.button("Download"):
        chat_data = json.dumps(st.session_state.chat_history, indent=4)
        st.sidebar.download_button("Download Chat", chat_data, file_name="chat_history.json", mime="application/json")


