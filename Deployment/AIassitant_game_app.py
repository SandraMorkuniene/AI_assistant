import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import PyPDF2
import io
import csv
from io import StringIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Function to process PDF and extract text
def process_pdf(uploaded_file):
    with io.BytesIO(uploaded_file.getvalue()) as byte_file:
        pdf_reader = PyPDF2.PdfReader(byte_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle None case
        return text

# Function to process plain text files
def process_text_file(uploaded_file):
    return uploaded_file.getvalue().decode("utf-8")

# Sidebar - Start new session button at the top
if st.sidebar.button("🆕 Start New Session"):
    st.session_state.clear()  # Reset everything
    st.rerun()  # Re-run the app to apply reset

# Sidebar for file upload
st.sidebar.header("📄 Upload Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDFs or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

# Sidebar - Model settings
st.sidebar.header("💡 Model Settings")
model_choice = st.sidebar.selectbox("Choose Model", ["gpt-3.5-turbo", "gpt-4"])
model_creativity = st.sidebar.slider("Model Creativity (Temperature)", 0.0, 1.0, 0.7, 0.1)
response_length_words = st.sidebar.slider("Response Length (Words)", 50, 500, 150, 10)
response_length_tokens = int(response_length_words * 0.75)

# Confirm settings button
if st.sidebar.button("Confirm Model Settings"):
    st.session_state.model_choice = model_choice
    st.session_state.model_creativity = model_creativity
    st.session_state.response_length_tokens = response_length_tokens
    st.session_state.model_confirmed = True
    st.success("Model settings confirmed.")

# Ensure session state variables exist
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "model_confirmed" not in st.session_state:
    st.session_state.model_confirmed = False

# Process uploaded documents if any
docs = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            text = process_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            text = process_text_file(uploaded_file)
        docs.append(text)

    if docs:
        embeddings = OpenAIEmbeddings()
        faiss_index = FAISS.from_texts(docs, embeddings)
        st.session_state.uploaded_files = faiss_index  # Store vector index
        st.success(f"Successfully indexed {len(docs)} documents.")
    else:
        st.error("No valid text found in the uploaded files.")

# Display conversation history
for message in st.session_state.conversation_history:
    st.chat_message(message["role"]).markdown(message["content"])

# User input (if settings are confirmed)
if st.session_state.model_confirmed:
    query = st.text_input("Ask a question:")
    
    if query:
        st.session_state.conversation_history.append({"role": "user", "content": query})

        # Retrieve relevant docs if uploaded
        if "uploaded_files" in st.session_state and st.session_state.uploaded_files:
            context = st.session_state.uploaded_files.similarity_search(query, k=2)
            context_text = "\n".join([doc.page_content for doc in context])
            prompt = f"Use the following context:\n{context_text}\nQuestion: {query}\nAnswer:"
        else:
            prompt = f"Answer the following question: {query}"

        # Construct message history
        messages = [SystemMessage(content="You are a helpful assistant.")]
        for message in st.session_state.conversation_history:
            if message["role"] == "user":
                messages.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                messages.append(AIMessage(content=message["content"]))

        messages.append(HumanMessage(content=prompt))

        # Get LLM response
        llm_response = llm(messages, temperature=st.session_state.model_creativity, max_tokens=st.session_state.response_length_tokens)

        # Add assistant response to history
        st.session_state.conversation_history.append({"role": "assistant", "content": llm_response.content})

        # Display response
        st.chat_message("assistant").markdown(llm_response.content)
else:
    st.warning("Confirm model settings before asking questions.")

# Function to save conversation as CSV
def save_conversation_csv():
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Role", "Message"])
    for message in st.session_state.conversation_history:
        writer.writerow([message["role"], message["content"]])
    return output.getvalue()

# Function to save conversation as PDF
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

# Sidebar - Save conversation options
st.sidebar.header("💾 Save Conversation")

if st.sidebar.button("Save as CSV"):
    csv_data = save_conversation_csv()
    st.sidebar.download_button("Download CSV", csv_data, "conversation.csv", mime="text/csv")

if st.sidebar.button("Save as PDF"):
    pdf_data = save_conversation_pdf()
    st.sidebar.download_button("Download PDF", pdf_data, "conversation.pdf", mime="application/pdf")
