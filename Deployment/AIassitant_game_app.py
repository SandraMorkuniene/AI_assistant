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
from langchain.schema import AIMessage
import boto3
import psycopg2

# Load secrets from Streamlit secrets
AWS_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = st.secrets["AWS_SECRET_KEY"]
BUCKET_NAME = st.secrets["BUCKET_NAME"]

DB_HOST = st.secrets["DB_HOST"]
DB_NAME = st.secrets["DB_NAME"]
DB_USER = st.secrets["DB_USER"]
DB_PASSWORD = st.secrets["DB_PASSWORD"]

# Initialize S3 Client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# Initialize Database Connection
def get_db_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=5432
    )

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
if 'system_prompt' not in st.session_state:
    st.session_state.system_prompt = ""

# Sidebar - API Key & Model Settings
st.sidebar.header("🔑 OpenAI API Key")
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
if st.sidebar.button("Confirm API Key"):
    st.session_state.openai_api_key = api_key
    st.session_state.api_confirmed = True
    st.sidebar.success("API Key Confirmed!")

st.sidebar.header("⚙️ Model Settings")
model_name = st.sidebar.selectbox("Select Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"], index=1)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)

# System Prompt Input
st.sidebar.header("📝 System Prompt")
st.session_state.system_prompt = st.sidebar.text_area("Enter a system prompt")

# New Chat Button
if st.sidebar.button("➕ New Chat"):
    st.session_state.chat_history = []
    st.session_state.total_cost = 0.0
    st.sidebar.success("New chat started!")
    st.rerun()

st.title("🤖 RAG-Enhanced Chatbot")
st.write("Ask questions based on uploaded documents!")

# Display Chat History
st.subheader("📜 Chat History")
for user, message in st.session_state.chat_history:
    st.write(f"**{user}:** {message}")

# User Input
user_input = st.text_input("You:", key="user_input")
if st.button("Send") and user_input.strip():
    with st.spinner("Processing..."):
        if st.session_state.vector_store is not None:
            retriever = st.session_state.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name=model_name, openai_api_key=st.session_state.openai_api_key), retriever=retriever)
            response = qa_chain.run(user_input)
        else:
            chat_model = ChatOpenAI(
                temperature=temperature,
                model_name=model_name,
                openai_api_key=st.session_state.openai_api_key
            )
            response = chat_model.invoke(user_input)
    
    response_text = response.content if isinstance(response, AIMessage) else str(response)

    # Token Cost Calculation
    prompt_tokens = len(user_input.strip().split()) * 1.33
    completion_tokens = len(response_text.split()) * 1.33
    total_tokens = prompt_tokens + completion_tokens
    cost = ((prompt_tokens / 1000) * MODEL_PRICING[model_name]["input"]) + ((completion_tokens / 1000) * MODEL_PRICING[model_name]["output"])
    st.session_state.total_cost += cost

    st.write(f"Cost for this response: ${cost:.6f}")
    st.write(f"Total Cost: ${st.session_state.total_cost:.6f}")

    # Update chat history
    st.session_state.chat_history.append(("You", user_input.strip()))
    st.session_state.chat_history.append(("Bot", response_text))
    st.rerun()
else:
    st.warning("Please enter and confirm your OpenAI API key to start chatting.")
