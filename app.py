import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
import re

st.set_page_config(page_title="RAG PDF Chatbot", page_icon="\U0001F4C4", layout="wide")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in environment. Please set it before running.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embedding_model()


def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def chunk_text(text, chunk_size=800, overlap=150):
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def build_faiss_index(chunks):
    embeddings = embed_model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    return index, embeddings


def retrieve_relevant_chunks(query, chunks, index, k=4):
    # Normalize case so "WHAT IS X" and "what is x" retrieve the same chunks
    query_normalized = query.strip().lower()
    query_embedding = embed_model.encode([query_normalized], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)
    system_prompt = (
        "You are a helpful assistant that answers questions strictly based on the "
        "provided PDF context. Treat uppercase, lowercase, and mixed-case questions "
        "identically \u2014 case never changes meaning or intent. If the answer isn't "
        "in the context, say you don't know based on the document, do not make things up."
    )
    user_prompt = (
        f"Context from PDF:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer clearly and concisely based only on the context above."
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return response.choices[0].message.content

st.title("\U0001F4C4 RAG-based PDF Chatbot")
st.caption("Powered by FAISS + Sentence-Transformers + Groq (Llama 3.3)")

if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "index" not in st.session_state:
    st.session_state.index = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

with st.sidebar:
    st.header("Upload your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None and uploaded_file.name != st.session_state.pdf_name:
        with st.spinner("Reading and indexing PDF... this may take a moment"):
            raw_text = extract_text_from_pdf(uploaded_file)
            if not raw_text.strip():
                st.error("Could not extract text from this PDF (it may be scanned/image-based).")
            else:
                chunks = chunk_text(raw_text)
                index, _ = build_faiss_index(chunks)
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chat_history = []
        if st.session_state.chunks:
            st.success(f"Indexed {len(st.session_state.chunks)} chunks from '{uploaded_file.name}'")

    if st.session_state.pdf_name:
        st.info(f"Active document: **{st.session_state.pdf_name}**")

    if st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()

if st.session_state.index is None:
    st.info("\U0001F448 Upload a PDF from the sidebar to get started.")
else:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask a question about your PDF (UPPER or lower case both work)...")

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                relevant_chunks = retrieve_relevant_chunks(
                    user_query, st.session_state.chunks, st.session_state.index
                )
                answer = generate_answer(user_query, relevant_chunks)
                st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
