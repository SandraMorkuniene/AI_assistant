import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings  # Update the import
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI  # Update the import
from langchain.schema import HumanMessage, SystemMessage, BaseMessage
import PyPDF2
import io
import csv
from io import StringIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

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

# Model settings sliders and selectors
model_choice = st.sidebar.selectbox("Choose Model", ["gpt-3.5-turbo", "gpt-4"])
model_creativity = st.sidebar.slider("Model Creativity (Temperature)", 0.0, 1.0, 0.7, 0.1)
response_length_words = st.sidebar.slider("Response Length (Words)", 50, 500, 150, 10)
response_length_tokens = int(response_length_words * 0.75)  # Approximate conversion: 1 word ≈ 0.75 tokens

# Add "Confirm" button to fix model settings
confirm_button = st.sidebar.button("Confirm Model Settings")

# Initialize session state for model settings if not present
if "model_choice" not in st.session_state:
    st.session_state.model_choice = model_choice
if "model_creativity" not in st.session_state:
    st.session_state.model_creativity = model_creativity
if "response_length_tokens" not in st.session_state:
    st.session_state.response_length_tokens = response_length_tokens
if "is_model_confirmed" not in st.session_state:
    st.session_state.is_model_confirmed = False  # Initial state is False

# Update the model settings when "Confirm" is pressed
if confirm_button:
    st.session_state.model_choice = model_choice
    st.session_state.model_creativity = model_creativity
    st.session_state.response_length_tokens = response_length_tokens
    st.session_state.is_model_confirmed = True  # Flag to confirm model settings
    st.success("Model settings confirmed for this session.")

# Place "Start New Session" Button in the top left corner
if st.sidebar.button("🆕 Start New Session"):
    # Reset session state to clear history, user input, uploaded files, and model confirmation
    st.session_state.conversation_history = []  # Clear conversation history
    st.session_state.user_input = ""  # Clear the text input when starting a new session
    st.session_state.uploaded_files = []  # Clear uploaded files
    st.session_state.is_model_confirmed = False  # Reset model confirmation flag

    # Reset other session state variables if necessary
    st.session_state.is_model_confirmed = False
    st.session_state.uploaded_files = []
    st.session_state.conversation_history = []
    st.session_state.user_input = ""

    st.info("Session reset! Please upload documents and confirm model settings again.")

# Initialize the conversation history in the session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""  # Initialize user input state
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []  # To store uploaded files

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
    
    # Vectorize and index the documents if they are uploaded
    if docs:
        embeddings = OpenAIEmbeddings()
        faiss_index = FAISS.from_texts(docs, embeddings)
        st.success(f"Successfully loaded and indexed {len(docs)} documents.")
        st.session_state.uploaded_files = docs  # Save docs to session state
    else:
        st.error("No documents found in the uploaded files.")
else:
    st.info("No documents uploaded. Chatbot will answer in a general way.")

# Display conversation history (chat-like interface)
if st.session_state.conversation_history:
    for i, message in enumerate(st.session_state.conversation_history):
        st.chat_message(message["role"]).markdown(message["content"])

# Get the user's query (if model settings are confirmed)
if st.session_state.is_model_confirmed:
    query = st.text_input("Ask a question:", value=st.session_state.user_input)

    if query:
        # Update the user input state
        st.session_state.user_input = query
        
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
            SystemMessage(content="You are a helpful assistant.")  # System message does not need type specified
        ]
        
        # Add conversation history to prompt
        for message in st.session_state.conversation_history:
            if message["role"] == "user":
                messages.append(HumanMessage(content=message["content"]))  # No need for 'type' here, it's inferred
            elif message["role"] == "assistant":
                messages.append(BaseMessage(content=message["content"]))  # No need for 'type' here either

        # Add current user query
        messages.append(HumanMessage(content=prompt))  # No need for 'type' here, it's inferred

        # Generate a response using the LLM with model creativity (temperature) and token count
        llm_response = llm(messages, temperature=st.session_state.model_creativity, max_tokens=st.session_state.response_length_tokens)

        # Add the model's response to the conversation history
        st.session_state.conversation_history.append({"role": "assistant", "content": llm_response.content})

        # Display the assistant's response in the chat
        st.chat_message("assistant").markdown(llm_response.content)
else:
    st.warning("Please confirm the model settings before asking questions.")

# Save the conversation to a CSV file
def save_conversation_csv():
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Role", "Message"])
    for message in st.session_state.conversation_history:
        writer.writerow([message["role"], message["content"]])
    return output.getvalue()

# Save the conversation to a PDF file
def save_conversation_pdf():
    output_pdf = io.BytesIO()
    c = canvas.Canvas(output_pdf, pagesize=letter)
    y_position = 750
    c.setFont("Helvetica", 10)
    
    for message in st.session_state.conversation_history:
        text = f"{message['role']}: {message['content']}"
        c.drawString(50, y_position, text)
        y_position -= 20
        if y_position < 50:
            c.showPage()
            y_position = 750
    c.save()
    
    output_pdf.seek(0)
    return output_pdf

# Provide download options for the user
st.sidebar.header("💾 Save Conversation")

if st.sidebar.button("Save as CSV"):
    csv_data = save_conversation_csv()
    st.sidebar.download_button("Download CSV", csv_data, "conversation.csv", mime="text/csv")

if st.sidebar.button("Save as PDF"):
    pdf_data = save_conversation_pdf()
    st.sidebar.download_button("Download PDF", pdf_data, "conversation.pdf", mime="application/pdf")
