import nest_asyncio
nest_asyncio.apply()

from dotenv import load_dotenv
import os
import streamlit as st
from rag_chain import build_chain, is_allowed

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY not found. Check your .env file.")

# Streamlit page config
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG Chatbot (LangChain + FAISS + Gemini)")

# Cache RAG chain to avoid rebuilding on every interaction
@st.cache_resource(show_spinner=False)
def get_chain():
    return build_chain()

chain, retriever = get_chain()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input from user
q = st.chat_input("Tanya soalan anda…")
if q:
    if not is_allowed(q):
        st.warning("Maaf, saya tidak dapat membantu dengan permintaan itu.")
    else:
        st.session_state.messages.append({"role": "user", "content": q})
    
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if q:
    with st.chat_message("assistant"):
        ans = chain.invoke(q)
        st.markdown(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})