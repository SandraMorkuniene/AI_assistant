import streamlit as st
import os
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import json

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
if 'api_confirmed' not in st.session_state:
    st.session_state.api_confirmed = False

# Sidebar - User API Key & Model Settings
st.sidebar.header("🔑 OpenAI API Key")
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
if st.sidebar.button("Confirm API Key"):
    st.session_state.openai_api_key = api_key
    st.session_state.api_confirmed = True
    st.sidebar.success("API Key Confirmed!")

st.sidebar.header("⚙️ Model Settings")
model_name = st.sidebar.selectbox("Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"], index=1)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
top_p = st.sidebar.slider("Top-p (nucleus sampling)", 0.0, 1.0, 1.0)
frequency_penalty = st.sidebar.slider("Frequency Penalty", -2.0, 2.0, 0.0)
presence_penalty = st.sidebar.slider("Presence Penalty", -2.0, 2.0, 0.0)

# Convert words to tokens (1 word ≈ 1.33 tokens)
words = st.sidebar.slider("Max Words", 50, 375, 150)  # Max tokens ≈ 500
max_tokens = int(words * 1.33)

# Download Chat History (Always Visible)
st.sidebar.subheader("⬇️ Download Chat")
export_format = st.sidebar.selectbox("Format", ["JSON", "CSV", "PDF"])
if st.sidebar.button("Download"):
    chat_data = json.dumps(st.session_state.chat_history, indent=4) if st.session_state.chat_history else "[]"
    st.sidebar.download_button("Download Chat", chat_data, file_name="chat_history.json", mime="application/json")

# Initialize OpenAI Chat Model
if st.session_state.api_confirmed and st.session_state.openai_api_key:
    chat_model = ChatOpenAI(
        temperature=temperature,
        model_name=model_name,
        openai_api_key=st.session_state.openai_api_key,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    memory = ConversationBufferMemory()

    st.title("🤖 Chatbot")
    st.write("Welcome! Start chatting with the bot below.")

    # Chat Interface
    user_input = st.text_input("You:", "")
    if st.button("Send") and user_input:
        with st.spinner("Thinking..."):
            response = chat_model.invoke(user_input)  # Directly call the model
            response_text = response.content  # Extract just the text response
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Bot", response_text))
            st.write(f"**Bot:** {response_text}")

    # Display Chat History
    st.subheader("📜 Chat History")
    for user, message in st.session_state.chat_history:
        st.write(f"**{user}:** {message}")
else:
    st.warning("Please enter and confirm your OpenAI API key to start chatting.")
