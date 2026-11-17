# RAG-based PDF Chatbot

Upload a PDF and ask questions about it. Built with Streamlit, PyMuPDF, FAISS,
Sentence-Transformers (all-MiniLM-L6-v2) and Groq.

## Run locally
```bash
pip install -r requirements.txt
export GROQ_API_KEY="your-key"
streamlit run app.py
```

## How it works
1. Extract text from the PDF (PyMuPDF)
2. Split into overlapping 800-character chunks
3. Embed chunks and index them with FAISS
4. Retrieve the top 4 chunks for a question
5. Send them as context to a Groq-hosted LLM to generate the answer

## Demo 
<img width="959" height="504" alt="Screenshot 2026-11-17 210205" src="https://github.com/user-attachments/assets/35d777b3-8657-4f5e-92d1-06524ffacd3a" />

<img width="959" height="500" alt="Screenshot 2026-11-17 210126" src="https://github.com/user-attachments/assets/c6a6cc0d-d487-4563-ab7e-9d392fb8b304" />

<img width="959" height="497" alt="Screenshot 2026-11-17 210250" src="https://github.com/user-attachments/assets/4231c8db-0197-474e-8641-0ee799e85809" />

Streamlit link - https://rag-based-pdf-chatbot-u6qzwq7effdxqznicnzbb8.streamlit.app/



