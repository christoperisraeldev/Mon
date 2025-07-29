import os
import json
import pickle
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

load_dotenv()

app = Flask(__name__)

# Configs
app.config['UPLOAD_FOLDER'] = 'knowledgebase_files'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Load embedding model once
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# In-memory vector store
vector_store = {"texts": [], "embeddings": None}
VECTOR_STORE_PATH = "vector_store.pkl"

# Perplexity API config
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"


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
    response = requests.post(PERPLEXITY_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data.get('choices', [{}])[0].get('message', {}).get('content', "No answer from API.")
    else:
        return f"Error from Perplexity API: {response.status_code} {response.text}"


def chunk_text(text, max_chunk_size=300):
    """Split text into chunks with max_chunk_size words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_chunk_size):
        chunk = " ".join(words[i:i + max_chunk_size])
        chunks.append(chunk)
    return chunks


def save_vector_store(store):
    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump(store, f)


def load_vector_store():
    if os.path.exists(VECTOR_STORE_PATH):
        with open(VECTOR_STORE_PATH, "rb") as f:
            return pickle.load(f)
    return {"texts": [], "embeddings": None}


def index_all_files():
    """Read text files, chunk, embed, and save vector store."""
    text_chunks = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if not filename.lower().endswith(('.txt', '.md')):
            continue
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            text_chunks.extend(chunk_text(text))
        except Exception as e:
            print(f"Error reading file {filename}: {e}")

    if not text_chunks:
        return {"texts": [], "embeddings": None}

    embeddings = embedding_model.encode(text_chunks, show_progress_bar=False)
    embeddings = np.array(embeddings)

    store = {"texts": text_chunks, "embeddings": embeddings}
    save_vector_store(store)
    return store


def search_knowledge_base(query, top_k=3, similarity_threshold=0.7):
    global vector_store
    # Load vector store if empty
    if not vector_store.get("texts"):
        vector_store_local = load_vector_store()
        if vector_store_local:
            vector_store.update(vector_store_local)
        else:
            return []

    if vector_store["embeddings"] is None or not vector_store["texts"]:
        return []

    query_embedding = embedding_model.encode([query])[0]

    # Normalize embeddings for cosine similarity
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    embeddings_norm = vector_store["embeddings"] / np.linalg.norm(vector_store["embeddings"], axis=1, keepdims=True)

    similarities = np.dot(embeddings_norm, query_norm)
    ranked_indices = np.argsort(similarities)[::-1]

    results = []
    for idx in ranked_indices:
        if similarities[idx] < similarity_threshold:
            continue
        if idx < len(vector_store["texts"]):
            results.append(vector_store["texts"][idx])
        if len(results) >= top_k:
            break
    return results


@app.route("/")
@app.route("/ai_chat")
def ai_chat():
    return render_template('ai_chat.html', active='ai_chat')


@app.route("/education")
def education():
    return render_template('education.html', active='education')


@app.route("/admin")
def admin():
    return render_template('admin.html', active='admin')


@app.route("/sources")
def sources():
    return render_template('sources.html', active='sources')


@app.route('/api/chat', methods=['POST'])
def api_chat():
    json_data = request.get_json(silent=True) or {}
    user_message = json_data.get("message", "")
    if user_message is None or user_message.strip() == "":
        return jsonify({"response": "Please enter a valid question."})
    user_message = user_message.strip()

    relevant_docs = search_knowledge_base(user_message, top_k=3)

    if relevant_docs:
        context = "\n\n---\n\n".join(relevant_docs)
        system_content = (
            "You are a medical assistant AI. **Answer ONLY using the knowledge base content below.** "
            "If the answer is not contained within the knowledge base, respond exactly: "
            "'No relevant information found in the knowledge base.' Do not guess or use any external info.\n\n"
            "Use markdown for formatting: headings, bullet points, tables, dosages, "
            "and continuous numbered references [1], [2], etc. Add a references section with real URLs "
            "from the knowledge base or credible sources. Do NOT make up references.\n\n"
            f"Knowledge Base Content:\n{context}\n\n---\n\nNow answer the question."
        )
        user_prompt = {"role": "user", "content": user_message}
        messages = [{"role": "system", "content": system_content}, user_prompt]
    else:
        system_content = (
            "You are a medical assistant AI. There is no relevant knowledge base content. "
            "Answer the question using your extensive medical knowledge and internet information only. "
            "Use markdown formatting with headings, bullet points, tables, dosages, "
            "and continuous numbered references [1], [2], etc. Include a references section with "
            "real verifiable URLs. Do NOT fabricate references."
        )
        user_prompt = {"role": "user", "content": user_message}
        messages = [{"role": "system", "content": system_content}, user_prompt]

    answer = call_perplexity_api(messages)

    return jsonify({"response": answer})


@app.route('/api/education/generate', methods=['POST'])
def generate_education():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    difficulty = data.get("difficulty", "easy")

    if not text or text.strip() == "":
        return jsonify({"flashcards": [], "mcqs": []})
    text = text.strip()

    system_prompt = {
        "role": "system",
        "content": (
            "You are an expert medical educator. Generate a list of flashcards and multiple choice questions (MCQs) "
            "based only on the provided text. For flashcards provide question and answer. For MCQs provide question, "
            "four options, and the correct answer. Make questions human-like, clear, relevant, and appropriate for the difficulty level "
            "(easy, medium, hard). Format response as JSON with keys 'flashcards' and 'mcqs'."
        )
    }
    user_prompt = {
        "role": "user",
        "content": f"Text:\n{text}\n\nDifficulty: {difficulty}"
    }

    messages = [system_prompt, user_prompt]
    result_text = call_perplexity_api(messages)

    flashcards = []
    mcqs = []

    try:
        result_json = json.loads(result_text)

        # Validate flashcards  
        raw_flashcards = result_json.get("flashcards", [])
        for fc in raw_flashcards:
            if (
                isinstance(fc, dict) and 
                isinstance(fc.get("question"), str) and 
                isinstance(fc.get("answer"), str)
            ):
                flashcards.append({
                    "question": fc["question"].strip(),
                    "answer": fc["answer"].strip()
                })

        # Validate MCQs  
        raw_mcqs = result_json.get("mcqs", [])
        for mcq in raw_mcqs:
            q = mcq.get("question")
            opts = mcq.get("options")
            corr = mcq.get("correct")

            if (
                isinstance(q, str) and q.strip() != "" and
                isinstance(opts, list) and len(opts) == 4 and
                all(isinstance(opt, str) for opt in opts) and
                isinstance(corr, str) and corr.strip() in opts
            ):
                mcqs.append({
                    "question": q.strip(),
                    "options": [opt.strip() for opt in opts],
                    "correct": corr.strip()
                })
            else:
                # Skip malformed MCQs
                pass

        if not mcqs:
            mcqs = [{
                "question": "What is the main topic of the text?",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct": "Option A"
            }]

    except json.JSONDecodeError:
        flashcards = [{
            "question": f"Summarize the text at {difficulty} difficulty.",
            "answer": text[:400] + ("..." if len(text) > 400 else "")
        }]
        mcqs = [{
            "question": "What is the main topic of the text?",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct": "Option A"
        }]

    return jsonify({"flashcards": flashcards, "mcqs": mcqs})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files['file']
    if file is None or file.filename is None or file.filename.strip() == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    global vector_store
    vector_store = index_all_files()

    return jsonify({"success": True, "filename": filename})


@app.route("/api/delete_file", methods=["POST"])
def api_delete_file():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename", "")
    if not filename:
        return jsonify({"error": "Filename is required"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        os.remove(file_path)

        global vector_store
        vector_store = index_all_files()

        return jsonify({"success": True})
    else:
        return jsonify({"error": "File not found"}), 404


@app.route("/api/files", methods=['GET'])
def api_files():
    try:
        files = os.listdir(app.config['UPLOAD_FOLDER'])
    except FileNotFoundError:
        files = []
    return jsonify({"files": files})


if __name__ == '__main__':
    # Load vector store on startup
    vector_store = load_vector_store()

    app.run(debug=True, host='0.0.0.0', port=5000)
