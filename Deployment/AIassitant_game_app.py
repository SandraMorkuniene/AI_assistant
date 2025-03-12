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

# OpenAI Pricing per 1K tokens (update if necessary)
MODEL_PRICING = {
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
}

# Initialize session state for vector DB
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# Sidebar for document upload
st.sidebar.header("📄 Upload Documents for Knowledge Base")
uploaded_files = st.sidebar.file_uploader("Upload PDFs or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

# Process uploaded documents
docs = []
if uploaded_files:
    for file in uploaded_files:
        if file.type == "application/pdf":
            loader = PyPDFLoader(file)
        else:
            loader = TextLoader(file)
        docs.extend(loader.load())  # Extract text

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # Embed & store in vector DB
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
    st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
    st.sidebar.success("Documents processed and stored in vector DB!")


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

    # Display Chat History Above Input Field
    st.subheader("📜 Chat History")
    for user, message in reversed(st.session_state.chat_history):
        st.write(f"**{user}:** {message}")

    # User Input
    user_input = st.text_input("You:", key="user_input")
    if st.button("Send") and user_input.strip():
        with st.spinner("Searching documents..."):

            # Retrieve relevant docs from vector DB
            retriever = st.session_state.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            qa_chain = RetrievalQA.from_chain_type(llm=chat_model, retriever=retriever)

            # Get response with retrieved context
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





# Initialize session state
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
def save_chat(export_format):
    if export_format == "JSON":
        chat_data = json.dumps(st.session_state.chat_history, indent=4)
        st.sidebar.download_button("Download Chat", chat_data, file_name="chat_history.json", mime="application/json")
    elif export_format == "CSV":
        df = pd.DataFrame(st.session_state.chat_history, columns=["User", "Message"])
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.sidebar.download_button("Download Chat", csv_data, file_name="chat_history.csv", mime="text/csv")
    elif export_format == "PDF":
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for user, message in st.session_state.chat_history:
            pdf.multi_cell(0, 10, f"{user}: {message}")
        pdf_output = "chat_history.pdf"
        pdf.output(pdf_output)
        with open(pdf_output, "rb") as f:
            st.sidebar.download_button("Download Chat", f, file_name="chat_history.pdf", mime="application/pdf")

st.sidebar.subheader("⬇️ Download Chat")
export_format = st.sidebar.selectbox("Format", ["JSON", "CSV", "PDF"])
if st.sidebar.button("Download"):
    save_chat(export_format)

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

    # Display Chat History Above Input Field
    st.subheader("📜 Chat History")
    for user, message in reversed(st.session_state.chat_history):  # Reverse order to show latest on top
        st.write(f"**{user}:** {message}")

    # Chat Interface
    user_input = st.text_input("You:", key="user_input")
    if st.button("Send") and user_input.strip():
        with st.spinner("Thinking..."):
            response = chat_model.invoke(user_input.strip())  # Directly call the model
            response_text = response.content  # Extract just the text response
            
            # Token Estimation & Cost Calculation
            prompt_tokens = len(user_input.strip().split()) * 1.33  # Approximation: 1 word ≈ 1.33 tokens
            completion_tokens = len(response_text.split()) * 1.33
            total_tokens = prompt_tokens + completion_tokens
            cost = ((prompt_tokens / 1000) * MODEL_PRICING[model_name]["input"]) + ((completion_tokens / 1000) * MODEL_PRICING[model_name]["output"])
            st.session_state.total_cost += cost
            
            # Update chat history
            st.session_state.chat_history.append(("You", user_input.strip()))
            st.session_state.chat_history.append(("Bot", response_text))
            
            st.write(f"💰 Estimated Cost: ${cost:.6f} (Total: ${st.session_state.total_cost:.6f})")
            
            st.session_state.user_input = ""  # Clear input field
            st.rerun()
        st.session_state.user_input = ""  # Ensure input field resets after response
else:
    st.warning("Please enter and confirm your OpenAI API key to start chatting.")
