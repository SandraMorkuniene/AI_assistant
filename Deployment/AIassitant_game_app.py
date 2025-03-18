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

# New Chat Button
if st.sidebar.button("➕ New Chat"):
    st.session_state.chat_history = []
    st.session_state.total_cost = 0.0
    st.sidebar.success("New chat started!")
    st.rerun()

st.sidebar.header("⚙️ Model Settings")
with st.sidebar.expander("🔽 Advanced Model Settings"):
    model_name = st.selectbox("Select Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"], index=1)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    top_p = st.slider("Top-p (nucleus sampling)", 0.0, 1.0, 1.0)
    frequency_penalty = st.slider("Frequency Penalty", -2.0, 2.0, 0.0)
    presence_penalty = st.slider("Presence Penalty", -2.0, 2.0, 0.0)

# System Prompt Input
st.sidebar.header("📝 System Prompt")
st.session_state.system_prompt = st.sidebar.text_area("Enter a system prompt")

# Sidebar for document upload
st.sidebar.header("📄 Upload Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDFs or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
if uploaded_files:
    docs = []
    for uploaded_file in uploaded_files:
        loader = PyPDFLoader(uploaded_file) if uploaded_file.type == "application/pdf" else TextLoader(uploaded_file)
        docs.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
    st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
    st.sidebar.success("✅ Documents stored in vector DB!")

st.title("🤖 RAG-Enhanced Chatbot")
st.write("Ask questions based on uploaded documents!")

st.subheader("📜 Chat History")
for user, message in st.session_state.chat_history:
    st.write(f"**{user}:** {message}")

user_input = st.text_input("You:", key="user_input")
if st.button("Send") and user_input.strip():
    with st.spinner("Processing..."):
        chat_model = ChatOpenAI(
            temperature=temperature,
            model_name=model_name,
            openai_api_key=st.session_state.openai_api_key,
            max_tokens=150,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
        response = chat_model.invoke(user_input)
    response_text = response.content if isinstance(response, AIMessage) else str(response)
    st.session_state.chat_history.append(("You", user_input.strip()))
    st.session_state.chat_history.append(("Bot", response_text))
    st.rerun()
else:
    st.warning("Please enter and confirm your OpenAI API key to start chatting.")
