import os
import json
import pickle
import traceback
from flask import (
    Flask, render_template, request, jsonify, send_from_directory,
    Response, stream_with_context, url_for
)
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import pytesseract
from PIL import Image
import whisper
import pdfplumber
from pdf2image import convert_from_path
import docx

from rake_nltk import Rake

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'knowledgebase_files'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Embedding model:
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Vector store local file path
VECTOR_STORE_PATH = "vector_store.pkl"
vector_store = {"texts": [], "embeddings": None, "sources": []}  # Store chunks & sources

# API keys and URLs from .env
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY")
GOOGLE_CSE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_CSE_CX")

SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")  # Optional

# Configure external tools paths (adjust if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
whisper_model = whisper.load_model("base")


# Download necessary nltk data
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


##########################
# Perplexity API helpers #
##########################


def call_perplexity_api_stream(messages):
    if PERPLEXITY_API_KEY is None:
        yield "Perplexity API key not set."
        return

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "sonar-pro",
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1500,
        "stream": True,
    }

    try:
        response = requests.post(PERPLEXITY_API_URL, headers=headers, json=payload, stream=True, timeout=60)
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith('data: '):
                data_str = decoded_line[len('data: '):]
                if data_str == '[DONE]':
                    break
                try:
                    data_json = json.loads(data_str)
                    token = data_json['choices'][0]['delta'].get('content')
                    if token:
                        yield token
                except Exception:
                    continue
    except Exception as e:
        yield f"\n\n[Error during streaming from Perplexity API: {str(e)}]\n\n"


def call_perplexity_api(messages):
    if PERPLEXITY_API_KEY is None:
        return "Perplexity API key not set."

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "sonar-pro",
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1500,
        "stream": False,
    }

    try:
        response = requests.post(PERPLEXITY_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get('choices', [{}])[0].get('message', {}).get('content', "No answer from API.")
    except Exception as e:
        return f"Error calling Perplexity API: {str(e)}"


#########################
# File processing utils  #
#########################

def chunk_text_with_source(text, source_title, max_chunk_words=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_chunk_words):
        chunk = " ".join(words[i:i + max_chunk_words])
        chunks.append((chunk, source_title))
    return chunks


def save_vector_store(store):
    try:
        with open(VECTOR_STORE_PATH, "wb") as f:
            pickle.dump(store, f)
    except Exception as e:
        print(f"Error saving vector store: {e}")


def load_vector_store():
    if os.path.exists(VECTOR_STORE_PATH):
        try:
            with open(VECTOR_STORE_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading vector store: {e}")
    return {"texts": [], "embeddings": None, "sources": []}


# Text extraction helpers

def extract_text_from_pdf_pdfplumber(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            pages_text = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages_text).strip()
    except Exception as e:
        print(f"PDF extract error: {e}")
        return ""


def extract_text_from_scanned_pdf(file_path):
    try:
        images = convert_from_path(file_path)
        text = "".join(pytesseract.image_to_string(image) for image in images)
        return text.strip()
    except Exception as e:
        print(f"Scanned PDF OCR error: {e}")
        return ""


def extract_text_with_ocr_fallback(file_path, min_text_length=50):
    text = extract_text_from_pdf_pdfplumber(file_path)
    if len(text.strip()) < min_text_length:
        text = extract_text_from_scanned_pdf(file_path)
    return text


def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs]
        return "\n".join(paragraphs).strip()
    except Exception as e:
        print(f"DOCX extract error: {e}")
        return ""


def index_all_files():
    text_chunks = []
    sources = []
    for fname in os.listdir(app.config['UPLOAD_FOLDER']):
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        ext = fname.lower().split('.')[-1]
        try:
            extracted_text = ""
            if ext in ['txt', 'md']:
                with open(fpath, 'r', encoding='utf-8') as f:
                    extracted_text = f.read()
            elif ext in ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif']:
                image = Image.open(fpath)
                extracted_text = pytesseract.image_to_string(image)
            elif ext == "pdf":
                extracted_text = extract_text_with_ocr_fallback(fpath)
            elif ext == 'docx':
                extracted_text = extract_text_from_docx(fpath)

            extracted_text = extracted_text.strip()
            if extracted_text:
                chunks_with_sources = chunk_text_with_source(extracted_text, fname)
                for chunk, source_title in chunks_with_sources:
                    text_chunks.append(chunk)
                    sources.append(source_title)
        except Exception as e:
            print(f"Error indexing file '{fname}': {e}")

    if not text_chunks:
        return {"texts": [], "embeddings": None, "sources": []}

    try:
        embeddings = embedding_model.encode(text_chunks, show_progress_bar=False)
        store = {"texts": text_chunks, "embeddings": np.array(embeddings), "sources": sources}
        save_vector_store(store)
        print(f"Indexed {len(text_chunks)} text chunks from uploaded files.")
        return store
    except Exception as e:
        print(f"Error during embedding: {e}")
        return {"texts": [], "embeddings": None, "sources": []}


def search_knowledge_base(query, top_k=3, threshold=0.5):
    global vector_store
    if not vector_store.get("texts"):
        vs_local = load_vector_store()
        if vs_local:
            vector_store.update(vs_local)
        else:
            return []
    if vector_store["embeddings"] is None or not vector_store["texts"]:
        return []

    try:
        q_emb = embedding_model.encode([query])
        q_emb = q_emb / np.linalg.norm(q_emb)
        norm_emb = vector_store["embeddings"] / np.linalg.norm(vector_store["embeddings"], axis=1, keepdims=True)
        sims = np.dot(norm_emb, q_emb).flatten()
    except Exception as e:
        print(f"Error computing embeddings similarity: {e}")
        return []

    rankings = np.argsort(sims)[::-1]
    results = []
    for i in rankings:
        if sims[i] < threshold:
            continue
        results.append({
            "text": vector_store["texts"][i],
            "score": float(sims[i]),
            "source_title": vector_store["sources"][i] if vector_store.get("sources") else "User Knowledgebase",
            "source_url": url_for("serve_uploaded_file", filename=vector_store["sources"][i]) if vector_store.get("sources") else "",
        })
        if len(results) >= top_k:
            break
    print(f"search_knowledge_base: Found {len(results)} chunks for query '{query}'.")
    return results


########################
# Keywords extraction  #
########################

def extract_keywords(text):
    try:
        rake = Rake()
        rake.extract_keywords_from_text(text)
        keywords = rake.get_ranked_phrases()
        return " ".join(keywords[:5])
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return text  # fallback entire


###########################
# External Search Helpers  #
###########################

def search_semantic_scholar(query, max_results=3):
    if not query:
        return []
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {"query": query, "limit": max_results, "fields": "title,abstract,url"}

        headers = {"User-Agent": "MON AI"}
        if SEMANTIC_SCHOLAR_API_KEY:
            headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        papers = r.json().get("data", [])
        results = []
        for p in papers:
            results.append({
                "text": f"{p.get('title', '')} - {p.get('abstract', '')}",
                "score": 0.7,
                "source_title": p.get('title', ''),
                "source_url": p.get("url", ""),
            })
        return results
    except Exception as e:
        print(f"Semantic Scholar API error: {e}")
        return []


def search_pubmed(query, max_results=3):
    if not query:
        return []
    try:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pubmed", "retmode": "json", "retmax": max_results, "term": query}
        r = requests.get(base_url, params=params, timeout=10)
        r.raise_for_status()

        idlist = r.json().get("esearchresult", {}).get("idlist", [])
        if not idlist:
            return []

        summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        summary_params = {"db": "pubmed", "retmode": "json", "id": ",".join(idlist)}
        resp = requests.get(summary_url, params=summary_params, timeout=10)
        resp.raise_for_status()

        results = []
        docs = resp.json().get("result", {})
        for pid in idlist:
            doc = docs.get(pid)
            if doc:
                results.append({
                    "text": doc.get("title", ""),
                    "score": 0.7,
                    "source_title": doc.get("title", ""),
                    "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
                })
        return results
    except Exception as e:
        print(f"PubMed API error: {e}")
        return []


def search_google_cse(query, max_results=3):
    if not query or not GOOGLE_CSE_API_KEY or not GOOGLE_CSE_SEARCH_ENGINE_ID:
        return []
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_CSE_API_KEY,
        "cx": GOOGLE_CSE_SEARCH_ENGINE_ID,
        "q": query,
        "num": max_results,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        items = r.json().get("items", [])
        results = []
        for item in items:
            results.append({
                "text": f"{item.get('title','')}: {item.get('snippet','')}",
                "score": 0.6,
                "source_title": item.get('title', ''),
                "source_url": item.get('link', ''),
            })
        return results
    except Exception as e:
        print(f"Google CSE API error: {e}")
        return []


###################
# Flask Endpoints #
###################


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.abspath(os.path.dirname(__file__)), 'favicon.ico')


@app.route('/knowledgebase_files/<path:filename>')
def serve_uploaded_file(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=False)
    except Exception as e:
        print(f"Error serving uploaded file {filename}: {e}")
        return "", 404


@app.route('/')
@app.route('/ai_chat')
def ai_chat():
    return render_template('ai_chat.html', active='ai_chat')


@app.route('/education')
def education():
    return render_template('education.html', active='education')


@app.route('/admin')
def admin():
    return render_template('admin.html', active='admin')


@app.route('/sources')
def sources():
    return render_template('sources.html', active='sources')


@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        data = request.get_json(silent=True) or {}
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"response": "Please enter a valid question."})

        print(f"User question: {user_message}")

        search_query = extract_keywords(user_message)
        print(f"Extracted keywords: {search_query}")

        # 1. User Knowledgebase semantic search
        kb_results = search_knowledge_base(user_message)
        print(f"User Knowledgebase results found: {len(kb_results)}")

        # 2. Trusted site web search (Google CSE)
        web_results = []
        try:
            web_results = search_google_cse(search_query)
            print(f"Google CSE search results count: {len(web_results)}")
        except Exception as e:
            print(f"Google CSE search error: {e}")

        # 3. Literature search (PubMed primary; else Semantic Scholar)
        lit_results = []
        try:
            pubmed_results = search_pubmed(search_query)
            if pubmed_results:
                lit_results = pubmed_results
            else:
                lit_results = search_semantic_scholar(search_query)
            print(f"Literature search results found: {len(lit_results)}")
        except Exception as e:
            print(f"Literature search error: {e}")
            lit_results = []

        # Tiered routing logic:
        if kb_results:
            selected_tier = "User Knowledgebase"
            selected_results = kb_results
        elif web_results:
            selected_tier = "Trusted Site Web Search"
            selected_results = web_results
        elif lit_results:
            selected_tier = "Scientific Literature"
            selected_results = lit_results
        else:
            selected_tier = "LLM Web (Perplexity)"
            selected_results = []

        # Prepare context text and references with inline links [1], [2], ...
        context_text = "\n\n".join([f"[{i+1}] {r.get('text', '')}" for i, r in enumerate(selected_results[:5])])
        references = []
        for idx, r in enumerate(selected_results):
            url = r.get("source_url") or ""
            title = r.get("source_title") or "Reference"
            if url.startswith("http"):
                references.append((title, url))
            elif selected_tier == "User Knowledgebase" and title:
                references.append((title, ""))  # No URL, just filename

        refs_text = "\n".join(f"[{i + 1}]: {url}" for i, (_, url) in enumerate(references) if url)

        if selected_results:
            system_prompt = (
                f"You are an expert medical assistant AI. Use the context below to answer the user's question "
                f"in a detailed, well-structured manner. Include explanations, drug dosing tables where relevant, "
                f"bullet points, clinical notes, and cite references with markdown [numbered] links.\n\n"
                f"Source tier: {selected_tier}\n\n"
                f"Context:\n{context_text}\n\nReferences:\n{refs_text}\n\n"
                f"Do not fabricate references, only use given citations."
            )
        else:
            system_prompt = (
                "You are a knowledgeable, professional medical assistant AI answering questions comprehensively. "
                "Include detailed explanations, dosing, structured text, and credible citations with URLs. "
                "Indicate your source tier as 'LLM Web (Perplexity)'. Do not fabricate citations."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {user_message}"}
        ]

        def generate():
            yield f"**Source**: {selected_tier}\n\n"
            yield from call_perplexity_api_stream(messages)

        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/api/education/generate', methods=['POST'])
def generate_education():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    difficulty = data.get("difficulty", "easy")
    if not text.strip():
        return jsonify({"flashcards": [], "mcqs": []})

    system_prompt = (
        "You are an expert medical educator. Based solely on the provided text, "
        "generate flashcards and multiple choice questions (MCQs). "
        "Format your response as a valid JSON with keys 'flashcards' and 'mcqs'. "
        "Each MCQ should have 'question', 'options' (list of strings), and a 'correct' field "
        "that exactly matches the correct option."
    )
    user_prompt = {"role": "user", "content": f"Text:\n{text}\n\nDifficulty: {difficulty}"}
    messages = [{"role": "system", "content": system_prompt}, user_prompt]

    try:
        result_text = call_perplexity_api(messages)
        result_json = json.loads(result_text)
        return jsonify({
            "flashcards": result_json.get("flashcards", []),
            "mcqs": result_json.get("mcqs", [])
        })
    except Exception:
        return jsonify({
            "flashcards": [{"question": "Summarize the text.", "answer": text[:400] + ("..." if len(text) > 400 else "")}],
            "mcqs": []
        })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file part in request"}), 400
    filename = file.filename
    if not filename or not filename.strip():
        return jsonify({"error": "No selected file"}), 400
    filename = secure_filename(filename.strip())
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(save_path)
    except Exception as e:
        return jsonify({"error": f"File save error: {e}"}), 500

    global vector_store
    vector_store = index_all_files()

    ext = os.path.splitext(filename)[1].lower()
    extracted_text = ""
    try:
        if ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']:
            extracted_text = pytesseract.image_to_string(Image.open(save_path))
        elif ext == '.pdf':
            extracted_text = extract_text_with_ocr_fallback(save_path)
        elif ext == '.docx':
            extracted_text = extract_text_from_docx(save_path)
        elif ext in ['.txt', '.md']:
            with open(save_path, 'r', encoding='utf-8') as f:
                extracted_text = f.read()
    except Exception as e:
        extracted_text = f"<Error extracting text: {e}>"

    return jsonify({"success": True, "filename": filename, "extracted_text": extracted_text})


@app.route("/api/delete_file", methods=["POST"])
def api_delete_file():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename", "")
    if not filename:
        return jsonify({"error": "Filename is required"}), 400
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            return jsonify({"error": f"Error deleting file: {e}"}), 500
        global vector_store
        vector_store = index_all_files()
        return jsonify({"success": True})
    else:
        return jsonify({"error": "File not found"}), 404


@app.route("/api/files", methods=['GET'])
def api_files():
    try:
        files = os.listdir(app.config['UPLOAD_FOLDER'])
    except Exception as e:
        print("File listing error:", e)
        files = []
    return jsonify({"files": files})


if __name__ == '__main__':
    vector_store = load_vector_store()
    app.run(debug=True, host='0.0.0.0', port=5000)
