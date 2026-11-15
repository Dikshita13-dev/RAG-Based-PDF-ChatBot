# 📄 RAG-based PDF Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload a PDF and ask natural language questions about its contents. The application extracts text from the uploaded document, generates embeddings, retrieves the most relevant sections using FAISS, and uses the Groq API with Llama 3.3 to generate context-aware answers.

## Features

* 📂 Upload any text-based PDF
* 🔍 Automatic text extraction using PyMuPDF
* ✂️ Intelligent text chunking with overlap
* 🧠 Semantic search using Sentence Transformers
* ⚡ Fast similarity search with FAISS
* 🤖 Answer generation using Groq (Llama 3.3 70B Versatile)
* 💬 Interactive chat interface built with Streamlit
* 🔄 Chat history support
* 🔤 Case-insensitive query handling

## Tech Stack

* Python
* Streamlit
* PyMuPDF
* Sentence Transformers (`all-MiniLM-L6-v2`)
* FAISS
* Groq API
* Llama 3.3 70B Versatile

## How It Works

1. Upload a PDF through the Streamlit interface.
2. The application extracts text from the PDF.
3. The extracted text is divided into overlapping chunks.
4. Each chunk is converted into an embedding using Sentence Transformers.
5. A FAISS index is built for efficient semantic retrieval.
6. When a user asks a question:

   * The query is embedded.
   * The most relevant chunks are retrieved from the FAISS index.
   * The retrieved context is passed to the Groq Llama 3.3 model.
7. The model generates an answer strictly based on the retrieved PDF context.

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
```

Install the required dependencies:

```bash
pip install streamlit pymupdf sentence-transformers faiss-cpu groq pyngrok
```

## Environment Variables

Set the following environment variables before running the application:

```text
GROQ_API_KEY=your_groq_api_key
NGROK_AUTH_TOKEN=your_ngrok_auth_token
```

Or enter them when prompted if you're running the Colab version.

## Run the Application

```bash
streamlit run app.py
```

If running in Google Colab, expose the Streamlit app using ngrok after starting the server.

## Project Structure

```
├── app.py
├── README.md
└── requirements.txt
```

## Limitations

* Works best with text-based PDFs.
* Scanned or image-only PDFs require OCR before they can be processed.
* Each uploaded PDF is indexed independently and is not persisted after the session ends.

## Future Improvements

* Support for multiple PDF uploads
* Persistent vector database (ChromaDB or Pinecone)
* Conversation memory
* Citation of source chunks
* Hybrid keyword + semantic retrieval
* OCR support for scanned PDFs
* Docker deployment
* Authentication for multiple users

## Demo
<img width="959" height="491" alt="Screenshot 2026-11-16 032845" src="https://github.com/user-attachments/assets/80b2579a-cd67-4577-9296-2e447face093" />
<img width="956" height="494" alt="Screenshot 2026-11-16 031542" src="https://github.com/user-attachments/assets/218e595d-ca81-4d47-882c-a1d5e30bd569" />

<img width="959" height="487" alt="Screenshot 2026-11-16 032332" src="https://github.com/user-attachments/assets/13719083-562d-4502-8ec4-801ca81059a0" />


You can run the application locally or in Google Colab using Streamlit and ngrok.
https://elixir-composite-waviness.ngrok-free.dev/

