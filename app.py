import os
import re

import faiss
import fitz  # PyMuPDF
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

APP_NAME = "DocuChat"
MODEL = "openai/gpt-oss-20b"
MAX_PDF_MB = 10
MAX_QUESTIONS_PER_SESSION = 30

st.set_page_config(page_title=f"{APP_NAME} · RAG PDF Chatbot", page_icon="📄", layout="wide")

# ---------------------------------------------------------------- styling
CSS = """
<style>
:root{
  --bg:#09050f; --side:#050507; --panel:#130a1c; --line:rgba(255,255,255,.08);
  --text:#ece7f3; --muted:#8d8499; --accent:#a855f7; --accent2:#7c3aed;
}
html, body, .stApp{ color:var(--text); font-family:"Inter","Segoe UI",system-ui,sans-serif; }
.stApp{
  background:
    radial-gradient(900px 520px at 78% 4%, rgba(124,58,237,.28), transparent 60%),
    radial-gradient(700px 480px at 20% 40%, rgba(88,28,135,.20), transparent 65%),
    var(--bg);
}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]{ visibility:hidden; height:0; }
[data-testid="stHeader"]{ background:transparent; }
.block-container{ max-width:980px; padding-top:1.2rem; padding-bottom:7rem; }

/* sidebar */
[data-testid="stSidebar"]{ background:var(--side); border-right:1px solid var(--line); }
[data-testid="stSidebar"] .block-container{ padding-top:1.4rem; }
.brand{ display:flex; align-items:center; gap:10px; font-weight:600; font-size:1.05rem; margin-bottom:1.1rem; }
.brand .logo{ width:30px; height:30px; border-radius:50%; background:#fff; color:#000;
  display:flex; align-items:center; justify-content:center; font-size:15px; }
.side-label{ color:var(--muted); font-size:.82rem; margin:1.1rem 0 .4rem; }
.doc-card{ background:rgba(168,85,247,.10); border:1px solid rgba(168,85,247,.35); border-radius:12px;
  padding:10px 12px; font-size:.86rem; word-break:break-word; }
.doc-card small{ color:var(--muted); display:block; margin-top:2px; }
.info-card{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px; margin-top:1.4rem; }
.info-card b{ font-size:.9rem; }
.info-card p{ color:var(--muted); font-size:.78rem; margin:.35rem 0 0; line-height:1.45; }

/* buttons */
.stButton > button, .stDownloadButton > button{
  width:100%; background:rgba(255,255,255,.05); color:var(--text); border:1px solid var(--line);
  border-radius:12px; padding:.5rem .9rem; font-size:.86rem; transition:border-color .15s, background .15s;
}
.stButton > button:hover, .stDownloadButton > button:hover{
  border-color:var(--accent); background:rgba(168,85,247,.14); color:#fff;
}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible{ outline:2px solid var(--accent); }
.stDownloadButton > button:disabled{ opacity:.4; }

/* file uploader */
[data-testid="stFileUploader"] section{
  background:var(--panel); border:1px dashed rgba(168,85,247,.45); border-radius:14px;
}
[data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span{ color:var(--muted); }

/* top bar pill */
.model-pill{ display:inline-flex; align-items:center; gap:8px; background:rgba(255,255,255,.05);
  border:1px solid var(--line); border-radius:999px; padding:.45rem .95rem; font-size:.84rem; }
.model-pill i{ width:8px; height:8px; border-radius:50%; background:#22c55e; display:inline-block; }

/* hero */
.hero{ text-align:center; padding:3.4rem 0 1.6rem; }
.orb{ width:86px; height:86px; margin:0 auto 1.6rem; border-radius:50%;
  background:radial-gradient(circle at 35% 30%, #f3e8ff 0%, #c084fc 22%, #7c3aed 55%, #2e1065 100%);
  box-shadow:0 0 70px rgba(168,85,247,.55), inset -8px -10px 24px rgba(0,0,0,.45);
  animation:float 5s ease-in-out infinite; }
@keyframes float{ 0%,100%{ transform:translateY(0);} 50%{ transform:translateY(-8px);} }
@media (prefers-reduced-motion: reduce){ .orb{ animation:none; } }
.hero h1{ font-size:2rem; font-weight:500; letter-spacing:-.01em; margin:0; padding:0; color:var(--text); }
.hero p{ color:var(--muted); margin:.6rem 0 0; font-size:.95rem; }

/* feature cards */
.cards{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-top:2.2rem; }
@media (max-width:760px){ .cards{ grid-template-columns:1fr; } }
.card{ background:rgba(255,255,255,.03); border:1px solid var(--line); border-radius:14px; padding:14px 16px; }
.card .row{ display:flex; justify-content:space-between; align-items:center; margin-bottom:.9rem; }
.card .ico{ width:28px; height:28px; border-radius:8px; background:rgba(255,255,255,.07);
  display:flex; align-items:center; justify-content:center; font-size:14px; }
.card .tag{ font-size:.7rem; background:rgba(255,255,255,.08); border-radius:999px; padding:.15rem .6rem; color:#d9d2e3; }
.card b{ font-size:.92rem; }
.card p{ color:var(--muted); font-size:.8rem; margin:.3rem 0 0; line-height:1.45; }

/* chat */
[data-testid="stChatMessage"]{ background:rgba(255,255,255,.04); border:1px solid var(--line);
  border-radius:16px; padding:.9rem 1rem; }
[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"]{ background:transparent; }
[data-testid="stChatInput"]{ background:var(--panel); border:1px solid rgba(168,85,247,.38); border-radius:18px; }
[data-testid="stChatInput"] textarea{ background:transparent; color:var(--text); }
[data-testid="stChatInput"] button{ background:linear-gradient(135deg,var(--accent),var(--accent2)); color:#fff; border-radius:50%; }
[data-testid="stAlert"]{ border-radius:12px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- RAG logic
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
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------- UI helpers
def transcript() -> str:
    lines = [f"{APP_NAME} chat — {st.session_state.pdf_name or 'no document'}", ""]
    for m in st.session_state.history:
        who = "You" if m["role"] == "user" else "Assistant"
        lines.append(f"{who}: {m['content']}\n")
    return "\n".join(lines)


def new_chat():
    st.session_state.chunks = None
    st.session_state.index = None
    st.session_state.history = []
    st.session_state.pdf_name = None
    st.session_state.questions = 0
    st.session_state.uploader_key += 1  # resets the file uploader


def main():
    api_key = get_api_key()
    if not api_key:
        st.error("GROQ_API_KEY is not configured. Add it in the app's Secrets settings.")
        st.stop()

    client = Groq(api_key=api_key)
    embed_model = load_embedding_model()

    defaults = {"chunks": None, "index": None, "history": [], "pdf_name": None,
                "questions": 0, "uploader_key": 0, "pending": None}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    # ------------------------------------------------------------ sidebar
    with st.sidebar:
        st.markdown(f'<div class="brand"><div class="logo">✦</div>{APP_NAME}</div>', unsafe_allow_html=True)
        if st.button("＋  New chat", key="new_chat"):
            new_chat()
            st.rerun()

        st.markdown('<div class="side-label">Document</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload a PDF", type="pdf", label_visibility="collapsed",
            key=f"uploader_{st.session_state.uploader_key}",
        )

        if uploaded_file is not None and uploaded_file.name != st.session_state.pdf_name:
            if uploaded_file.size > MAX_PDF_MB * 1024 * 1024:
                st.error(f"File too large. Please upload a PDF under {MAX_PDF_MB} MB.")
            else:
                with st.spinner("Reading and indexing PDF..."):
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

        if st.session_state.pdf_name:
            st.markdown(
                f'<div class="doc-card">📄 {st.session_state.pdf_name}'
                f'<small>{len(st.session_state.chunks)} chunks indexed</small></div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="side-label">Session</div>', unsafe_allow_html=True)
            st.caption(f"Questions used: {st.session_state.questions} / {MAX_QUESTIONS_PER_SESSION}")
            if st.button("Clear chat history", key="clear"):
                st.session_state.history = []
                st.session_state.questions = 0
                st.rerun()

        st.markdown(
            '<div class="info-card"><b>How it works</b>'
            "<p>Your PDF is split into chunks, embedded with MiniLM, searched with FAISS, "
            "and answered by an LLM on Groq using only those passages.</p></div>",
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------ top bar
    left, right = st.columns([5, 1])
    with left:
        st.markdown(f'<span class="model-pill"><i></i>{MODEL} · Groq</span>', unsafe_allow_html=True)
    with right:
        st.download_button(
            "Export chat", data=transcript(), file_name="chat.txt", mime="text/plain",
            disabled=not st.session_state.history,
        )

    # ------------------------------------------------------------ empty state
    if st.session_state.index is None:
        st.markdown(
            '<div class="hero"><div class="orb"></div>'
            "<h1>Ready to chat with your PDF?</h1>"
            "<p>Upload a document in the sidebar and ask anything about it.</p></div>"
            '<div class="cards">'
            '<div class="card"><div class="row"><span class="ico">📝</span><span class="tag">Summarize</span></div>'
            "<b>Quick summaries</b><p>Get the main points of a long report in seconds.</p></div>"
            '<div class="card"><div class="row"><span class="ico">🔎</span><span class="tag">Search</span></div>'
            "<b>Find answers</b><p>Ask in plain language and get answers from your own document.</p></div>"
            '<div class="card"><div class="row"><span class="ico">✅</span><span class="tag">Grounded</span></div>'
            "<b>No guessing</b><p>If the document doesn't say it, the assistant tells you so.</p></div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ------------------------------------------------------------ chat state
    if not st.session_state.history:
        st.markdown(
            '<div class="hero" style="padding-bottom:1rem"><div class="orb"></div>'
            f"<h1>Ask anything about {st.session_state.pdf_name}</h1></div>",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        if c1.button("Summarize this document", key="chip1"):
            st.session_state.pending = "Summarize this document in a few short paragraphs."
        if c2.button("List the key points", key="chip2"):
            st.session_state.pending = "List the key points of this document as short bullets."
        if c3.button("What is this about?", key="chip3"):
            st.session_state.pending = "What is this document about and who is it for?"

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask anything about your PDF...")
    if st.session_state.pending and not user_query:
        user_query = st.session_state.pending
    st.session_state.pending = None
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
    st.rerun()


main()
