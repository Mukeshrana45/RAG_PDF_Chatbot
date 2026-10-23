# app.py — Flask server with Groq-powered RAG backend

from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv  # loads .env file into os.environ automatically
import os
import tempfile
from rag_engine import load_and_index_pdf, build_qa_chain

load_dotenv()  # reads .env file and sets GROQ_API_KEY in environment — call before anything else

app = Flask(__name__, static_folder="static")

qa_chain  = None
retriever = None


@app.route("/")
def index():
    """Serve the main HTML UI."""
    return send_from_directory("static", "index.html")


@app.route("/upload", methods=["POST"])
def upload_pdf():
    """POST /upload — indexes the uploaded PDF and prepares QA chain."""
    global qa_chain, retriever

    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"success": False, "message": "Empty filename"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"success": False, "message": "Only PDF files accepted"}), 400

    try:
        # Save to temp file so LangChain's PyPDFLoader can read it from disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        # Build FAISS index (embedding — fast, local)
        vector_store = load_and_index_pdf(tmp_path)

        # Build Groq-powered QA chain
        qa_chain, retriever = build_qa_chain(vector_store)

        os.unlink(tmp_path)  # clean up temp file

        return jsonify({"success": True, "message": "PDF indexed successfully!"})

    except ValueError as e:
        # Catches missing GROQ_API_KEY with a helpful message
        return jsonify({"success": False, "message": str(e)}), 400

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask_question():
    """POST /ask — runs RAG pipeline and returns answer."""
    global qa_chain, retriever

    if qa_chain is None:
        return jsonify({"success": False, "message": "Please upload a PDF first"}), 400

    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"success": False, "message": "No question provided"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"success": False, "message": "Question cannot be empty"}), 400

    try:
        # Run Groq-powered RAG chain — returns answer in ~0.5s
        answer = qa_chain.invoke(question)

        # Fetch source page numbers for citation display
        source_docs = retriever.invoke(question)
        pages = sorted(set(
            int(doc.metadata.get("page", 0)) + 1
            for doc in source_docs
            if "page" in doc.metadata
        ))

        return jsonify({"success": True, "answer": answer, "pages": pages})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    print("\n🚀 RAG PDF Chatbot running at: http://127.0.0.1:5000\n")
    # HuggingFace Spaces uses port 7860
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))