import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
import PyPDF2
import io

# Initialize LLM (in case no documents are uploaded)
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Function to process PDF and extract text
def process_pdf(uploaded_file):
    with io.BytesIO(uploaded_file.getvalue()) as byte_file:
        pdf_reader = PyPDF2.PdfReader(byte_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

# Function to process plain text files
def process_text_file(uploaded_file):
    return uploaded_file.getvalue().decode("utf-8")

# Sidebar for model settings and document upload
st.sidebar.header("📄 Upload Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDFs or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

# Sidebar for model settings
st.sidebar.header("💡 Model Settings")
model_choice = st.sidebar.selectbox("Choose Model", ["gpt-3.5-turbo", "gpt-4"])
model_creativity = st.sidebar.slider("Model Creativity (Temperature)", 0.0, 1.0, 0.7, 0.1)

# Slider to control the response length in words (convert to tokens internally)
response_length_words = st.sidebar.slider("Response Length (Words)", 50, 500, 150, 10)
response_length_tokens = int(response_length_words * 0.75)  # Approximate conversion: 1 word ≈ 0.75 tokens

# Button to start a new session
if st.sidebar.button("🆕 Start New Session"):
    st.session_state.conversation_history = []  # Clear conversation history

# Initialize the conversation history in the session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Process uploaded documents if any
docs = []  # This will hold your document text
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            text = process_pdf(uploaded_file)
            docs.append(text)
        elif uploaded_file.type == "text/plain":
            text = process_text_file(uploaded_file)
            docs.append(text)
    
    # Vectorize and index the documents
    if docs:
        embeddings = OpenAIEmbeddings()
        faiss_index = FAISS.from_texts(docs, embeddings)
        st.success(f"Successfully loaded and indexed {len(docs)} documents.")
    else:
        st.error("No documents found in the uploaded files.")
else:
    st.info("No documents uploaded. Chatbot will answer in a general way.")

# Display conversation history (chat-like interface)
if st.session_state.conversation_history:
    for i, message in enumerate(st.session_state.conversation_history):
        st.chat_message(message["role"]).markdown(message["content"])

# Get the user's query
query = st.text_input("Ask a question:")

if query:
    # Add the user's query to the conversation history
    st.session_state.conversation_history.append({"role": "user", "content": query})
    
    # If documents are provided, use FAISS for retrieval
    if uploaded_files:
        context = faiss_index.similarity_search(query, k=2)  # Fetch top 2 relevant docs
        context_text = "\n".join([doc.page_content for doc in context])  # Extract page_content from each document
        prompt = f"Use the following context to answer the question:\n{context_text}\nQuestion: {query}\nAnswer:"
    else:
        prompt = f"Answer the following question in a general way: {query}"

    # Prepare messages for LLM
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=prompt)
    ]

    # Generate a response using the LLM with model creativity (temperature) and token count
    llm_response = llm(messages, temperature=model_creativity, max_tokens=response_length_tokens)

    # Add the model's response to the conversation history
    st.session_state.conversation_history.append({"role": "assistant", "content": llm_response.content})

    # Display the assistant's response in the chat
    st.chat_message("assistant").markdown(llm_response.content)

