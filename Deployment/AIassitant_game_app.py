import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import PyPDF2
import io
import csv
from io import StringIO

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

def process_pdf(uploaded_file):
    with io.BytesIO(uploaded_file.getvalue()) as byte_file:
        pdf_reader = PyPDF2.PdfReader(byte_file)
        text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
    return text

def process_text_file(uploaded_file):
    return uploaded_file.getvalue().decode("utf-8")

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None
if "model_confirmed" not in st.session_state:
    st.session_state.model_confirmed = False
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

if st.sidebar.button("🆕 Start New Session"):
    st.session_state.clear()
    st.rerun()

st.sidebar.header("📄 Upload Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDFs or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

if uploaded_files and (st.session_state.uploaded_files is None or len(uploaded_files) != len(st.session_state.uploaded_files)):
    with st.spinner("Processing documents..."):
        docs = [process_pdf(f) if f.type == "application/pdf" else process_text_file(f) for f in uploaded_files]
        embeddings = OpenAIEmbeddings()
        faiss_index = FAISS.from_texts(docs, embeddings)
        st.session_state.uploaded_files = faiss_index
    st.success(f"Successfully indexed {len(docs)} documents.")

st.sidebar.header("💡 Model Settings")
st.session_state.model_choice = st.sidebar.selectbox("Choose Model", ["gpt-3.5-turbo", "gpt-4"], index=0)
st.session_state.model_creativity = st.sidebar.slider("Model Creativity (Temperature)", 0.0, 1.0, 0.7, 0.1)
st.session_state.response_length_words = st.sidebar.slider("Response Length (Words)", 50, 500, 150, 10)

if st.sidebar.button("Confirm Model Settings"):
    st.session_state.model_confirmed = True
    st.success("Model settings confirmed.")

for message in st.session_state.conversation_history:
    st.chat_message(message["role"]).markdown(message["content"])

if st.session_state.model_confirmed:
    query = st.text_input("Ask a question:", value=st.session_state.user_input)

    if query:
        st.session_state.user_input = ""  # Clear input field
        st.session_state.conversation_history.append({"role": "user", "content": query})

        context_text = ""
        if st.session_state.uploaded_files:
            context = st.session_state.uploaded_files.similarity_search(query, k=2)
            context_text = "\n".join([doc.page_content for doc in context])
        
        prompt = (f"Use this context:\n{context_text}\nQuestion: {query}\n"
                  f"Please provide a concise response within {st.session_state.response_length_words} words.")
        
        messages = [SystemMessage(content="You are a helpful assistant.")]
        for msg in st.session_state.conversation_history:
            messages.append(HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=prompt))
        
        llm_response = llm(messages, temperature=st.session_state.model_creativity, max_tokens=512)
        
        st.session_state.conversation_history.append({"role": "assistant", "content": llm_response.content})
        st.chat_message("assistant").markdown(llm_response.content)
else:
    st.warning("Confirm model settings before asking questions.")

def save_conversation_csv():
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Role", "Message"])
    for msg in st.session_state.conversation_history:
        writer.writerow([msg["role"], msg["content"]])
    return output.getvalue()

st.sidebar.header("💾 Download Conversation")
st.sidebar.download_button("Download CSV", save_conversation_csv(), "conversation.csv", "text/csv")
