import os
import tempfile
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import openai
import streamlit as st
from langchain_community.document_loaders import TextLoader

# 1️⃣ Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings()

# 2️⃣ Initialize FAISS index for storing document embeddings
faiss_index = FAISS.from_documents([], embeddings)

# Function to process PDF or TXT into chunks
def process_document(file):
    if file.type == "application/pdf":
        loader = PyPDFLoader(file)
    elif file.type == "text/plain":
        loader = TextLoader(file)
    else:
        return []

    docs = loader.load()
    return docs

# Function to add documents to FAISS index
def add_to_faiss(docs):
    # Embedding and adding documents to FAISS index
    docs_with_embeddings = embeddings.embed_documents([doc.page_content for doc in docs])
    faiss_index.add(docs_with_embeddings)

# Function to retrieve the most relevant chunks using FAISS
def retrieve_relevant_chunks(query):
    # Get the most relevant documents from FAISS
    query_embedding = embeddings.embed_query(query)
    results = faiss_index.similarity_search(query_embedding, k=5)  # Retrieve top 5 results
    return results

# Function to get an answer using the most relevant documents
def get_answer_from_documents(query):
    relevant_docs = retrieve_relevant_chunks(query)
    chain = load_qa_chain(openai, chain_type="stuff")
    answer = chain.run(input_documents=relevant_docs, question=query)
    return answer

# Fallback function for answering general queries
def get_answer_without_documents(query):
    # If documents aren't provided or the question doesn't require documents, use a fallback method
    # You can use OpenAI's API or a static response (predefined knowledge) here
    prompt = f"Answer this question: {query}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0].text.strip()

# Sidebar for file upload
st.sidebar.header("📄 Upload Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDFs or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

# 3️⃣ Add documents to the FAISS index if uploaded
if uploaded_files:
    docs = []
    for uploaded_file in uploaded_files:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf" if uploaded_file.type == "application/pdf" else ".txt") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name
        
        # Process the document
        document_chunks = process_document(temp_file_path)
        docs.extend(document_chunks)
        
    # Add documents to the FAISS index
    add_to_faiss(docs)

    # Display total documents loaded
    st.write(f"Total documents loaded: {len(docs)}")

# Input query from user
query = st.text_input("Ask a question:")

if query:
    if uploaded_files:
        # If documents are uploaded, use them to answer the query
        answer = get_answer_from_documents(query)
    else:
        # If no documents are uploaded, use general knowledge or a pre-trained model
        answer = get_answer_without_documents(query)
    
    st.write("Answer:", answer)
