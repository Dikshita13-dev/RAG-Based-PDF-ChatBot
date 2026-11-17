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

<img width="959" height="497" alt="image" src="https://github.com/user-attachments/assets/1f1bc90d-8029-4654-a902-f3e69c60cb93" />

<img width="959" height="494" alt="image" src="https://github.com/user-attachments/assets/c849264b-ddc6-46a0-8134-ff93e7814380" />

Streamlit Link - https://rag-based-pdf-chatbot-6wzt4bt7zhjxu3bmerqnqy.streamlit.app/




