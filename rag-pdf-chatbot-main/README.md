# 🤖 RAG PDF Chatbot

> Ask questions over **any PDF document** using Retrieval-Augmented Generation (RAG)

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-yellow?logo=huggingface)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-orange)
![Groq](https://img.shields.io/badge/Groq-LPU%20Inference-red)
![Flask](https://img.shields.io/badge/Flask-Backend-lightgrey)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🧠 What is RAG? (Simple Explanation)

Imagine you have a 100-page PDF. You can't paste the whole thing into ChatGPT. RAG solves this:

```
Your PDF
   │
   ▼
Split into small chunks (500 characters each)
   │
   ▼
Convert each chunk into a number vector (semantic embedding)
   │
   ▼
Store all vectors in FAISS (a local vector database)
   │
   ▼
You ask a question
   │
   ▼
Find the 4 most similar chunks using vector search
   │
   ▼
Send those chunks + your question to an LLM
   │
   ▼
LLM generates an answer based ONLY on those chunks
```

The LLM never sees the full PDF — it only sees the most relevant pieces.
This is why it's fast, accurate, and doesn't hallucinate (make things up).

---

## ⚠️ Honest Limitations (Read This First!)

> This section is important. RAG is powerful but it has real limitations.
> If you test this chatbot and it says "I don't know", that's not always a bug —
> it's often just how RAG works. Here's what to expect:

### ❌ What this chatbot CANNOT do

| Limitation                                    | Why it happens                                      | Example                                                 |
| --------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------- |
| **Cannot describe images**                    | RAG only reads text, not pixels                     | "What does the diagram show?" → "I don't know"          |
| **Cannot read scanned PDFs well**             | Scanned PDFs are photos, not text                   | A photographed paper document → garbled output          |
| **Cannot answer from memory**                 | It only knows what's in YOUR PDF                    | "Who is Elon Musk?" → "Not in this document"            |
| **Cannot answer cross-document questions**    | Only one PDF loaded at a time                       | "Compare this with that other file" → won't work        |
| **May miss answers on large PDFs**            | Retrieves only top 4 chunks out of hundreds         | Answer might be in chunk 50 but chunk 12 gets retrieved |
| **Stylized/decorative fonts confuse OCR**     | Tesseract reads design PDFs poorly                  | Presentation PDFs with gradient text → garbled          |
| **Cannot summarize very long PDFs perfectly** | "Summarize everything" needs all chunks, not just 4 | Better to ask specific questions                        |

### ✅ What this chatbot IS good at

- PDFs with real embedded text (research papers, reports, books, resumes)
- Specific factual questions ("Who won?", "What was the conclusion?", "How many X?")
- Finding information buried deep in a long document
- Extracting key points from a section

### 💡 Tips for best results

- Ask **specific questions** instead of vague ones
  - ❌ "Tell me everything"
  - ✅ "What were the main findings in section 3?"
- Use **text-heavy PDFs** — reports, papers, ebooks, resumes work best
- If an answer seems wrong, try **rephrasing the question**
- PDFs exported from Word/Google Docs work better than design tools like Canva

---

## 🏗️ Architecture

```
PDF File (text-based)
   │
   ▼
PyPDFLoader → extracts text per page
   │
   ▼
RecursiveCharacterTextSplitter → 500 char chunks, 50 char overlap
   │
   ▼
HuggingFace Embeddings (all-MiniLM-L6-v2, local, free)
   │
   ▼
FAISS Vector Store (local vector database)
   │
   ▼
User asks a question
   │
   ├── FAISS retrieves top 4 relevant chunks
   │
   └── Groq API (llama-3.1-8b-instant) generates answer
            │
            ▼
       Answer + Source Pages shown in UI
```

---

## ⚙️ Tech Stack

| Component           | Tool                           | Why chosen                                     |
| ------------------- | ------------------------------ | ---------------------------------------------- |
| PDF text extraction | PyPDFLoader                    | Reads embedded fonts cleanly, no OCR noise     |
| Text splitting      | RecursiveCharacterTextSplitter | Preserves sentence context at chunk boundaries |
| Embeddings          | `all-MiniLM-L6-v2`             | Free, 80MB, runs on CPU, high quality          |
| Vector database     | FAISS (CPU)                    | Local, no cloud needed, fast similarity search |
| LLM                 | Groq `llama-3.1-8b-instant`    | ~0.5s responses vs 30s+ for local CPU models   |
| Backend             | Flask                          | Lightweight Python web server                  |
| Frontend            | Custom HTML/CSS/JS             | Animated dark UI, no framework needed          |

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Aaricacoding/rag-pdf-chatbot.git
cd rag-pdf-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get a free Groq API key

1. Go to **https://console.groq.com**
2. Sign up (free) → API Keys → Create API Key
3. Copy your key

### 4. Create your `.env` file

```
GROQ_API_KEY=your_key_here
```

> ⚠️ Never commit this file to GitHub. It's already in `.gitignore`.

### 5. Run

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## 📖 How to Use

1. **Upload** a text-based PDF (research paper, report, resume, ebook)
2. Click **Index PDF** — wait for ✅ (takes 5–15 seconds for embedding)
3. **Ask a specific question** about the document
4. Get an answer with **source page numbers** cited

---

## 📁 Project Structure

```
rag-pdf-chatbot/
├── app.py              # Flask server — API endpoints + serves HTML
├── rag_engine.py       # RAG pipeline — PDF loading, FAISS indexing, QA chain
├── requirements.txt    # Python dependencies
├── .env.example        # Template for your API key (copy to .env)
├── .gitignore          # Keeps .env and cache out of git
├── README.md           # This file
└── static/
    └── index.html      # Animated frontend UI
```

---

## 🔧 Customization

| What to change             | Where               | How                                                    |
| -------------------------- | ------------------- | ------------------------------------------------------ |
| Number of chunks retrieved | `rag_engine.py`     | Change `k=4` in `as_retriever()`                       |
| Chunk size                 | `rag_engine.py`     | Adjust `chunk_size=500`                                |
| LLM model                  | `rag_engine.py`     | Replace `llama-3.1-8b-instant` with another Groq model |
| UI colors/theme            | `static/index.html` | Edit CSS variables at the top of the file              |

---

## 🏷️ Skills Demonstrated

- **RAG architecture** — end-to-end implementation from scratch
- **Vector databases** — FAISS indexing and semantic similarity search
- **LangChain 0.3** — LCEL chains, retrievers, prompt templates
- **HuggingFace** — local embedding models
- **Groq API** — production LLM inference integration
- **Flask** — REST API backend with file upload handling
- **Frontend** — animated UI without any JS framework

---

## 📄 License

MIT License — free to use, modify, and distribute.
