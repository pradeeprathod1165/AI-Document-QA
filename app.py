import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

st.title("AI Document Question Answering")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Processing document..."):

        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        splitter = CharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        texts = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        db = FAISS.from_documents(texts, embeddings)

    st.success("Document processed!")

    query = st.text_input("Ask a question about the document")

    if query:

        with st.spinner("Thinking..."):

            docs = db.similarity_search(query, k=3)

            context = "\n\n".join([doc.page_content for doc in docs])

            llm = Ollama(model="phi3")

            prompt = f"""
You are an AI assistant that answers questions based on the given document.

Context:
{context}

Question:
{query}

Give a clear and helpful answer.
"""

            answer = llm.invoke(prompt)

            st.write("### Answer")
            st.write(answer)