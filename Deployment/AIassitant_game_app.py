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
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# OpenAI Pricing per 1K tokens
MODEL_PRICING = {
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
}

# Initialize session state variables
if 'qdrant_client' not in st.session_state:
    st.session_state.qdrant_client = None
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

if uploaded_files:
    for uploaded_file in uploaded_files:
        # 1️⃣ Save file locally before processing
        file_path = os.path.join("/tmp", uploaded_file.name)  # Change "/tmp" for Windows if needed
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())  # Write the file to disk

        # 2️⃣ Load the file using PyPDFLoader or TextLoader
        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(file_path)  # Now using the saved file path
        else:
            loader = TextLoader(file_path)

        docs = loader.load()

        # 3️⃣ Process the document (e.g., vectorization)
        st.success(f"✅ Successfully loaded: {uploaded_file.name}")

# Qdrant Client Setup (AWS)
QDRANT_URL = "http://your-ec2-ip:6333"  # Replace with your Qdrant server IP and port
QDRANT_COLLECTION = "documents"  # Name of the Qdrant collection
qdrant_client = QdrantClient(url=QDRANT_URL)

# Initialize OpenAI Embeddings for LangChain
embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)

# Function to add documents to Qdrant
def add_documents_to_qdrant(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # Convert text to embeddings
    vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])

    # Prepare the documents for Qdrant
    points = [
        PointStruct(
            id=f"{i}",
            vector=vector.tolist(),
            payload={"text": chunk.page_content}
        )
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    
    # Store the documents in Qdrant
    qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    st.sidebar.success("✅ Documents stored in Qdrant!")

# Step 4: Convert and store documents in Qdrant
if uploaded_files:
    docs = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(uploaded_file)
            docs.extend(loader.load())
        elif uploaded_file.type == "text/plain":
            loader = TextLoader(uploaded_file)
            docs.extend(loader.load())

    add_documents_to_qdrant(docs)

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
    for user, message in st.session_state.chat_history:
        st.write(f"**{user}:** {message}")

    # User Input
    user_input = st.text_input("You:", key="user_input")
    if st.button("Send") and user_input.strip():
        with st.spinner("Processing..."):
            if qdrant_client:
                # Use Qdrant retriever to fetch relevant documents for RAG
                retriever = qdrant_client.search(
                    collection_name=QDRANT_COLLECTION,
                    query_vector=embeddings.embed_query(user_input),
                    limit=3
                )

                # Build the retrieval chain
                qa_chain = RetrievalQA.from_chain_type(llm=chat_model, retriever=retriever)
                response = qa_chain.run(user_input)
            else:
                # If no documents exist, use OpenAI model directly
                response = chat_model.invoke(user_input)

        # Token Cost Calculation
        prompt_tokens = len(user_input.strip().split()) * 1.33
        completion_tokens = len(response.split()) * 1.33
        total_tokens = prompt_tokens + completion_tokens
        cost = ((prompt_tokens / 1000) * MODEL_PRICING[model_name]["input"]) + ((completion_tokens / 1000) * MODEL_PRICING[model_name]["output"])

        if 'total_cost' not in st.session_state:
            st.session_state.total_cost = 0

        st.session_state.total_cost += cost

        st.write(f"Cost for this response: ${cost:.6f}")
        st.write(f"Total Cost: ${st.session_state.total_cost:.6f}")

        # Update chat history
        st.session_state.chat_history.append(("You", user_input.strip()))
        st.session_state.chat_history.append(("Bot", response))

        st.rerun()
else:
    st.warning("Please enter and confirm your OpenAI API key to start chatting.")
