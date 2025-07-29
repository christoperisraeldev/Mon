# Backend Explanation and Code Snippets

## 1. `/api/chat` — AI Chat Answering Endpoint

### Purpose
Handles user question input, searches uploaded knowledge base documents for relevant content, then:

- If relevant document snippets found: instruct AI to answer *only* using these snippets.
- If no relevant docs: AI answers from general medical knowledge or the internet.
- Ensures formatted markdown response with real, continuous numbered references.

### Key Backend Logic Outline

```python
@app.route('/api/chat', methods=['POST'])
def api_chat():
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"response": "Please enter a valid question."})

    relevant_docs = search_knowledge_base(user_message, top_k=3)  # semantic search from uploaded docs

    if relevant_docs:
        context = "\n\n---\n\n".join(relevant_docs)
        system_content = (
            "You are a medical assistant AI. Answer ONLY using the knowledge base content below. "
            "If the answer is not contained there, respond exactly: 'No relevant information found in the knowledge base.' "
            "Do NOT guess or use external info. Format your answer in markdown with headings, bullet points, tables, "
            "dosages, and continuous numbered references [1], [2], etc. Include a references section with real URLs "
            "only from the knowledge base or credible sources.\n\n"
            f"Knowledge Base Content:\n{context}\n\n---\n\nNow answer the question."
        )
    else:
        system_content = (
            "You are a medical assistant AI. No relevant knowledge base content present. "
            "Answer the question using your medical and internet knowledge, formatted with markdown, headings, bullet points, "
            "tables, dosages, and number references with real URLs. Do NOT fabricate references."
        )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_message}
    ]

    answer = call_perplexity_api(messages)

    return jsonify({"response": answer})
```

### Notes
- `search_knowledge_base` uses embeddings + cosine similarity to return best matching text chunks from documents.
- The system prompt **forces strict usage of knowledge base content if available**, else fallback.
- Continuous numbering of references and real URL citation is enforced per prompt.
- Markdown formatting is expected in the returned string for rich display.

## 2. `/api/upload` — Upload Knowledge Base Documents

### Purpose
Allow uploading `.txt` or `.md` files to expand the knowledge base.

### Backend Snippet

```python
@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    # Rebuild embeddings index after upload to include new content
    global vector_store
    vector_store = index_all_files()

    return jsonify({"success": True, "filename": filename})
```

## 3. `/api/delete_file` — Delete Knowledge Base Document

### Purpose
Remove a previously uploaded file and update the knowledge base index accordingly.

### Backend Snippet

```python
@app.route('/api/delete_file', methods=['POST'])
def api_delete_file():
    data = request.get_json()
    filename = data.get("filename", "")
    if not filename:
        return jsonify({"error": "Filename is required"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        os.remove(file_path)

        # Rebuild vector store after deletion
        global vector_store
        vector_store = index_all_files()

        return jsonify({"success": True})
    else:
        return jsonify({"error": "File not found"}), 404
```

## 4. `/api/education/generate` — Generate Flashcards and MCQs

### Purpose
Given input text & difficulty level, generate educational flashcards and MCQs via AI.

### Backend Snippet

```python
@app.route('/api/education/generate', methods=['POST'])
def generate_education():
    data = request.get_json()
    text = data.get("text", "").strip()
    difficulty = data.get("difficulty", "easy")

    if not text:
        return jsonify({"flashcards": [], "mcqs": []})

    system_prompt = {
        "role": "system",
        "content": (
            "You are an expert medical educator. Generate flashcards and multiple choice questions (MCQs) "
            "based on the provided text. For flashcards, provide question and answer. For MCQs, provide question, "
            "four options, and the correct answer. Make them clear, relevant, and appropriate for "
            f"the {difficulty} difficulty level. Respond in JSON format with 'flashcards' and 'mcqs' keys."
        )
    }
    user_prompt = {
        "role": "user",
        "content": f"Text:\n{text}\n\nDifficulty: {difficulty}"
    }

    messages = [system_prompt, user_prompt]
    result_text = call_perplexity_api(messages)

    try:
        result_json = json.loads(result_text)
        flashcards = result_json.get("flashcards", [])
        mcqs = result_json.get("mcqs", [])
    except Exception:
        # Fallback summary flashcard and simple MCQ if AI response is malformed
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
```

# Frontend Explanation and Code Snippets

## Chat Page (`ai_chat.html`)

### Core Interaction

- User types question in textarea.
- On clicking "Send", frontend calls `/api/chat` POST with `{ message: "" }`.
- Response markdown answer is rendered below.
- Input auto-resizes as user types.

### Example JS snippet sending a chat message:

```js
async function sendMessage() {
    const userInput = document.getElementById('chat-input').value.trim();
    if (!userInput) return;

    // Add user question to chat display (implement as needed)
    appendUserMessage(userInput);

    // Send to backend
    const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userInput })
    });
    const data = await response.json();

    // Render markdown answer (using markdown-it or similar)
    const answerHtml = markdownIt.render(data.response);
    appendAIMessage(answerHtml);

    // Clear input and keep focus
    document.getElementById('chat-input').value = '';
}
```

## Admin Page (`admin.html`)

### File Upload

- User selects file (`.txt` or `.md`).
- Clicking "Upload" sends file in POST `/api/upload`.
- Files list updated using `/api/files` GET.
- Each file can be deleted with POST `/api/delete_file`.

### Example file upload JS snippet:

```js
document.getElementById('upload-file-btn').addEventListener('click', async () => {
    const fileInput = document.getElementById('file-upload');
    if (fileInput.files.length === 0) {
        alert('Please select a file.');
        return;
    }
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
    });
    const result = await response.json();
    if (result.success) {
        alert('File uploaded.');
        fileInput.value = '';
        loadFilesList();  // function to refresh displayed files
    } else {
        alert('Upload failed: ' + (result.error || 'Unknown error'));
    }
});
```

## Education Page (`education.html`)

### MCQ & Flashcard Generation

- User inputs a text passage and selects difficulty.
- Submission sends POST `/api/education/generate` with input text and difficulty.
- AI response JSON with flashcards and MCQs displayed.
- MCQ answers remain hidden initially until user selects option and clicks “Check Answer.”

# Summary

| Feature              | Backend Endpoint           | Frontend Interaction/Elements                      |
|----------------------|---------------------------|--------------------------------------------------|
| AI chat answering    | POST `/api/chat`           | Chat input textarea, Send button, markdown display |
| KB file upload       | POST `/api/upload`         | File selector, Upload button, Files list display |
| KB file delete       | POST `/api/delete_file`    | Delete buttons next to file names                 |
| Education generation | POST `/api/education/generate` | Text input, difficulty selector, Generate button, Flashcards & MCQ display  |