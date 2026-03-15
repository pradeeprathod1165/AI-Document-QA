import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

st.set_page_config(page_title="AI Document Chat", layout="wide")

st.title("📚 AI Document Chat")

FAISS_PATH = "faiss_index"

# -----------------------------
# Load LLM once (faster + token limit)
# -----------------------------
if "llm" not in st.session_state:
    st.session_state.llm = Ollama(
        model="phi3:mini",
        num_predict=120
    )

# -----------------------------
# Cache embeddings
# -----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# -----------------------------
# Chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.header("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    process = st.button("Process Documents")

    if process and uploaded_files:

        all_documents = []

        for uploaded_file in uploaded_files:

            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.read())

            loader = PyPDFLoader(uploaded_file.name)
            docs = loader.load()

            all_documents.extend(docs)

        st.write("✂️ Splitting documents...")

        splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        texts = splitter.split_documents(all_documents)

        st.write("🧠 Creating embeddings...")

        db = FAISS.from_documents(texts, embeddings)

        db.save_local(FAISS_PATH)

        st.session_state.db = db

        st.success("✅ Documents processed and saved!")

    if st.button("Clear Chat"):
        st.session_state.messages = []

# -----------------------------
# Load FAISS if exists
# -----------------------------
if "db" not in st.session_state:

    if os.path.exists(FAISS_PATH):

        st.session_state.db = FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

# -----------------------------
# Chat Interface
# -----------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])

# -----------------------------
# User Input
# -----------------------------
query = st.chat_input("Ask a question about your documents")

if query and "db" in st.session_state:

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):

        thinking = st.status("🔎 Searching documents...", expanded=False)

        docs = st.session_state.db.similarity_search(query, k=1)

        thinking.update(label="🧠 Generating answer...", state="running")

        # Limit context size (major speed improvement)
        context = docs[0].page_content[:1000]

        prompt = f"""
Use the context to answer the question.

Context:
{context}

Question: {query}

Answer in 2-3 short sentences.
"""

        response = st.session_state.llm.stream(prompt)

        message_placeholder = st.empty()

        full_response = ""

        for chunk in response:
            full_response += chunk
            message_placeholder.write(full_response)

        thinking.update(label="✅ Done", state="complete")

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )