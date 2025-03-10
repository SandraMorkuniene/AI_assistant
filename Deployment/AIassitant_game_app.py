
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import speech_recognition as sr
import queue
import tempfile
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import os
import re
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="Hello! I am your AI assistant. How can I help you today?")]

st.sidebar.header("Settings")
openai_api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
model_name = st.sidebar.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"])

st.title("Chatbot with Text & Voice Input 🎤")

if not openai_api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=temperature, model_name=model_name)

# Display previous chat messages
for message in st.session_state.messages:
    if isinstance(message, AIMessage):
        st.chat_message("assistant").markdown(message.content)
    elif isinstance(message, HumanMessage):
        st.chat_message("user").markdown(message.content)

# Queue for audio data
audio_queue = queue.Queue()

def process_audio(audio_data):
    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_data)
        temp_audio_path = temp_audio.name

    with sr.AudioFile(temp_audio_path) as source:
        audio = recognizer.record(source)
        try:
            voice_text = recognizer.recognize_google(audio)
            st.session_state.transcribed_text = voice_text
            st.success(f"Recognized speech: {voice_text}")
        except sr.UnknownValueError:
            st.error("Could not understand the audio")
        except sr.RequestError:
            st.error("Error with speech recognition service")

    os.remove(temp_audio_path)

# Streamlit WebRTC Component for Real-Time Speech Input
webrtc_ctx = webrtc_streamer(
    key="speech-to-text",
    mode=WebRtcMode.SENDRECV,
    audio_receiver_size=1024,
    async_processing=True
)

if webrtc_ctx.audio_receiver:
    audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
    for frame in audio_frames:
        audio_queue.put(frame.to_ndarray().tobytes())
        process_audio(frame.to_ndarray().tobytes())

# Show transcribed text in chat input
user_input = st.text_input("Type your message or use voice input", value=st.session_state.get("transcribed_text", ""))

if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").markdown(user_input)

    response = llm([HumanMessage(content=user_input)])

    st.session_state.messages.append(response)
    st.chat_message("assistant").markdown(response.content)

    # Clear transcribed text after sending
    st.session_state.transcribed_text = ""



