# rag_engine.py — RAG pipeline with smart PDF loading + Groq API
# Strategy: try PyPDFLoader first (clean text), fall back to OCR only if needed

import os
from langchain_community.document_loaders import PyPDFLoader        # clean text extraction
from langchain_text_splitters import RecursiveCharacterTextSplitter  # chunk splitter
from langchain_huggingface import HuggingFaceEmbeddings              # local embeddings
from langchain_community.vectorstores import FAISS                   # vector database
from langchain_groq import ChatGroq                                  # Groq fast LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def load_and_index_pdf(pdf_path: str) -> FAISS:
    """
    Smart PDF loader — uses PyPDFLoader (clean, fast, accurate).
    OCR is only needed for truly scanned PDFs with zero embedded text.
    Your PDFs have embedded text so PyPDFLoader gives much cleaner results.
    """

    print("[INFO] Loading PDF with PyPDFLoader...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()  # each page → Document(page_content, metadata)

    # Check how much text was actually extracted
    total_text = sum(len(doc.page_content) for doc in documents)
    print(f"[INFO] Extracted {total_text} characters from {len(documents)} pages")

    # If PDF has very little text (scanned/image-only), warn the user
    if total_text < 100:
        raise ValueError(
            "This PDF has almost no extractable text. "
            "It may be a scanned document. OCR support coming soon."
        )

    # Split into 500-char chunks with 50-char overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"[INFO] Split into {len(chunks)} chunks")

    # Embed with all-MiniLM-L6-v2 — fast, local, no API key
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # Build FAISS index
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("[INFO] FAISS index ready")

    return vector_store


def build_qa_chain(vector_store: FAISS):
    """
    Builds LCEL chain: FAISS retriever + Groq Llama 3.1 LLM.
    Returns (chain, retriever).
    """

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Add it to your .env file. Get free key at https://console.groq.com"
        )

    # Groq LLM — llama-3.1-8b-instant: fast, free, capable
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,           # deterministic answers
        groq_api_key=groq_api_key
    )

    # Fetch top 4 chunks for more context per question
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # Prompt — answer only from context, be natural and helpful
    prompt = PromptTemplate.from_template(
        "You are a helpful assistant analyzing a document.\n"
        "Answer the question using ONLY the information in the context below.\n"
        "If the answer is not in the context, say 'This information is not in the document.'\n"
        "Be clear, concise, and natural.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    def format_docs(docs):
        # Join chunks with separator so LLM sees clear boundaries
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    # LCEL chain
    chain = (
        {
            "context":  retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

    # Pre-load embedding model at startup so first user request isn't slow
print("[INFO] Pre-loading embedding model...")
HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
print("[INFO] Embedding model ready")