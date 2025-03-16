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
if uploaded_files:
    docs = []  # Store document texts

    conn = get_db_connection()  # Connect to AWS RDS
    cur = conn.cursor()

    for uploaded_file in uploaded_files:
        # 🔹 Step 1: Save file in S3
        s3_file_key = f"uploads/{uploaded_file.name}"
        s3_client.upload_fileobj(uploaded_file, BUCKET_NAME, s3_file_key)
        s3_file_url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{s3_file_key}"

        # 🔹 Step 2: Save metadata in AWS RDS (PostgreSQL)
        cur.execute(
            "INSERT INTO documents (file_name, s3_url) VALUES (%s, %s) RETURNING id",
            (uploaded_file.name, s3_file_url)
        )
        doc_id = cur.fetchone()[0]  # Get the document ID
        conn.commit()

        # 🔹 Step 3: Load document content for vector storage
        temp_file_path = os.path.join("/tmp", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(temp_file_path)
        else:
            loader = TextLoader(temp_file_path)

        docs.extend(loader.load())  # Load document content
        os.remove(temp_file_path)  # Delete temp file

    # 🔹 Step 4: Convert to vectors & store in FAISS
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
    st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)

    st.sidebar.success("✅ Documents stored in vector DB!")

    cur.close()
    conn.close()  # Close RDS connection
    
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

