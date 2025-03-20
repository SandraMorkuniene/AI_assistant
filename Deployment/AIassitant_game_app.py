import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
import PyPDF2
import io

# Initialize LLM (in case no documents are uploaded)
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 1️⃣ Sidebar for document upload
st.sidebar.header("📄 Upload Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDFs or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

docs = []  # This will hold your document text

# 2️⃣ Process uploaded documents (if any)
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Process and add documents to `docs` list
        if uploaded_file.type == "application/pdf":
            text = process_pdf(uploaded_file)
            docs.append(text)
        elif uploaded_file.type == "text/plain":
            text = process_text_file(uploaded_file)
            docs.append(text)
    
    # 3️⃣ If documents are provided, vectorize them with OpenAI Embeddings and FAISS
    if docs:
        embeddings = OpenAIEmbeddings()
        faiss_index = FAISS.from_texts(docs, embeddings)  # Notice how we pass the list of text strings to FAISS
        st.success(f"Successfully loaded and indexed {len(docs)} documents.")
    else:
        st.error("No documents found in the uploaded files.")
else:
    st.info("No documents uploaded. Chatbot will answer in a general way.")

# 4️⃣ Get the user's query
query = st.text_input("Ask a question:")

if query:
    if uploaded_files:
        # Use FAISS to retrieve context if documents were uploaded
        context = faiss_index.similarity_search(query, k=2)  # Fetch top 2 relevant docs
        context_text = "\n".join([doc for doc in context])  # Combine document text into one string

        # Combine the context with the user query for the LLM model
        prompt = f"Use the following context to answer the question:\n{context_text}\nQuestion: {query}\nAnswer:"

        # Generate a response using the LLM
        response = llm(prompt)
        st.write(response['choices'][0]['message']['content'])
    else:
        # If no documents, just use the LLM's general response
        prompt = f"Answer the following question in a general way: {query}"
        response = llm(prompt)
        st.write(response['choices'][0]['message']['content'])

# Helper functions to process PDFs and text files

def process_pdf(uploaded_file):
    # Extract text from a PDF file
    with io.BytesIO(uploaded_file.getvalue()) as byte_file:
        pdf_reader = PyPDF2.PdfReader(byte_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

def process_text_file(uploaded_file):
    # Extract text from a plain text file
    return uploaded_file.getvalue().decode("utf-8")
