# AI Document QA

An AI application that allows users to upload a PDF and ask questions about the document.

## Features
- Upload PDF
- Ask questions
- AI-generated answers
- Local LLM using Ollama

## Tech Stack
- Streamlit
- LangChain
- FAISS
- Ollama
- HuggingFace Embeddings

## Installation

git clone https://github.com/pradeeprathod1165/AI-Document-QA.git

cd AI-Document-QA

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

streamlit run app.py
