import os
import re

import faiss
import fitz  # PyMuPDF
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

MAX_PDF_MB = 10
MAX_QUESTIONS_PER_SESSION = 30

st.set_page_config(page_title="RAG PDF Chatbot", page_icon="📄", layout="wide")


def get_api_key() -> str:
    """Read the key from Streamlit secrets (cloud) or env var (local)."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text


def chunk_text(text, chunk_size=800, overlap=150):
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def build_faiss_index(embed_model, chunks):
    embeddings = embed_model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    return index


def retrieve_relevant_chunks(embed_model, query, chunks, index, k=4):
    q = embed_model.encode([query.strip().lower()], convert_to_numpy=True).astype("float32")
    _, indices = index.search(q, k)
    return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]


def generate_answer(client, query, context_chunks):
    context = "\n\n".join(context_chunks)
    system_prompt = (
        "You are a helpful assistant that answers questions strictly based on the "
        "provided PDF context. Treat uppercase, lowercase, and mixed-case questions "
        "identically — case never changes meaning or intent. If the answer isn't "
        "in the context, say you don't know based on the document, do not make things up."
    )
    user_prompt = (
        f"Context from PDF:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer clearly and concisely based only on the context above."
    )
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return response.choices[0].message.content


def main():
    api_key = get_api_key()
    if not api_key:
        st.error("GROQ_API_KEY is not configured. Add it in the app's Secrets settings.")
        st.stop()

    client = Groq(api_key=api_key)
    embed_model = load_embedding_model()

    st.title("📄 RAG-based PDF Chatbot")
    st.caption("Powered by FAISS + Sentence-Transformers + Groq")

    defaults = {"chunks": None, "index": None, "history": [], "pdf_name": None, "questions": 0}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    with st.sidebar:
        st.header("Upload your PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None and uploaded_file.name != st.session_state.pdf_name:
            if uploaded_file.size > MAX_PDF_MB * 1024 * 1024:
                st.error(f"File too large. Please upload a PDF under {MAX_PDF_MB} MB.")
            else:
                with st.spinner("Reading and indexing PDF... this may take a moment"):
                    try:
                        raw_text = extract_text_from_pdf(uploaded_file.getvalue())
                    except Exception:
                        raw_text = ""
                        st.error("Could not open this PDF. Is it corrupted or password-protected?")
                    if raw_text and not raw_text.strip():
                        st.error("No text found (the PDF may be scanned/image-based).")
                    elif raw_text:
                        chunks = chunk_text(raw_text)
                        st.session_state.index = build_faiss_index(embed_model, chunks)
                        st.session_state.chunks = chunks
                        st.session_state.pdf_name = uploaded_file.name
                        st.session_state.history = []
                        st.session_state.questions = 0
                        st.success(f"Indexed {len(chunks)} chunks from '{uploaded_file.name}'")

        if st.session_state.pdf_name:
            st.info(f"Active document: **{st.session_state.pdf_name}**")

        if st.button("Clear chat history"):
            st.session_state.history = []
            st.session_state.questions = 0
            st.rerun()

    if st.session_state.index is None:
        st.info("👈 Upload a PDF from the sidebar to get started.")
        return

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask a question about your PDF...")
    if not user_query:
        return

    if st.session_state.questions >= MAX_QUESTIONS_PER_SESSION:
        st.warning("Question limit reached for this session. Click 'Clear chat history' to continue.")
        return

    st.session_state.history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                chunks = retrieve_relevant_chunks(
                    embed_model, user_query, st.session_state.chunks, st.session_state.index
                )
                answer = generate_answer(client, user_query, chunks)
            except Exception:
                answer = "Sorry, something went wrong while contacting the model. Please try again."
            st.markdown(answer)

    st.session_state.history.append({"role": "assistant", "content": answer})
    st.session_state.questions += 1


main()
