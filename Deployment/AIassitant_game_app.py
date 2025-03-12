import streamlit as st
import os
import json
import pandas as pd
from fpdf import FPDF
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# OpenAI Pricing per 1K tokens
MODEL_PRICING = {
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
}

# Initialize session state variables
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
if 'api_confirmed' not in st.session_state:
    st.session_state.api_confirmed = False
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
if 'total_cost' not in st.session_state:
    st.session_state.total_cost = 0.0

# Sidebar - API Key & Model Settings
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

# Convert words to tokens
words = st.sidebar.slider("Max Words", 50, 375, 150)
max_tokens = int(words * 1.33)

# Sidebar for document upload
st.sidebar.header("📄 Upload Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDFs or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

# Process uploaded documents
docs = []
if uploaded_files:
    for file in uploaded_files:
        if file.type == "application/pdf":
            loader = PyPDFLoader(file)
        else:
            loader = TextLoader(file)
        docs.extend(loader.load())
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    
    # Embed & store in vector DB
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
    st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
    st.sidebar.success("Documents processed and stored in vector DB!")

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
    
    st.title("🤖 RAG-Enhanced Chatbot")
    st.write("Ask questions based on uploaded documents!")
    
    # Display Chat History
    st.subheader("📜 Chat History")
    for user, message in reversed(st.session_state.chat_history):
        st.write(f"**{user}:** {message}")
    
    # User Input
    user_input = st.text_input("You:", key="user_input")
    if st.button("Send") and user_input.strip():
        with st.spinner("Searching documents..."):
            retriever = st.session_state.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            qa_chain = RetrievalQA.from_chain_type(llm=chat_model, retriever=retriever)
            response = qa_chain.run(user_input)
            
            # Token Cost Calculation
            prompt_tokens = len(user_input.strip().split()) * 1.33
            completion_tokens = len(response.split()) * 1.33
            total_tokens = prompt_tokens + completion_tokens
            cost = ((prompt_tokens / 1000) * MODEL_PRICING[model_name]["input"]) + ((completion_tokens / 1000) * MODEL_PRICING[model_name]["output"])
            st.session_state.total_cost += cost
            
            # Update chat history
            st.session_state.chat_history.append(("You", user_input.strip()))
            st.session_state.chat_history.append(("Bot", response))
            
            st.write(f"💰 Estimated Cost: ${cost:.6f} (Total: ${st.session_state.total_cost:.6f})")
            
            st.session_state.user_input = ""
            st.rerun()
else:
    st.warning("Please enter and confirm your OpenAI API key to start chatting.")

