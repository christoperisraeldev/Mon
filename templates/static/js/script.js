document.addEventListener('DOMContentLoaded', () => {
    // AI Chat elements
    const chatWindow = document.getElementById('chat-window');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const imageUpload = document.getElementById('image-upload');

    function appendMessage(sender, text) {
        const div = document.createElement('div');
        div.classList.add(sender === 'AI' ? 'ai-message' : 'user-message');
        div.textContent = text;
        chatWindow.appendChild(div);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    function updateLastAIMessage(newText) {
        const aiMessages = chatWindow.querySelectorAll('.ai-message');
        if (!aiMessages.length) return;
        aiMessages[aiMessages.length - 1].textContent = newText;
    }

    if (sendBtn) {
        sendBtn.addEventListener('click', async () => {
            const message = chatInput.value.trim();
            if (!message) {
                alert('Please enter a message.');
                return;
            }

            appendMessage('User', message);
            chatInput.value = '';

            if (imageUpload && imageUpload.files.length > 0) {
                appendMessage('AI', 'Image upload feature not yet implemented in this demo.');
                imageUpload.value = '';
                return;
            }

            appendMessage('AI', 'Thinking...');

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });
                const data = await response.json();
                updateLastAIMessage(data.response);
            } catch (e) {
                updateLastAIMessage('Error communicating with server.');
            }
        });
    }

    // Education elements
    const generateEduBtn = document.getElementById('generate-edu');
    const educationText = document.getElementById('education-text');
    const difficultySelect = document.getElementById('difficulty');
    const educationOutput = document.getElementById('education-output');

    if (generateEduBtn) {
        generateEduBtn.addEventListener('click', async () => {
            const text = educationText.value.trim();
            if (!text) {
                alert('Please enter some text to generate questions.');
                return;
            }
            const difficulty = difficultySelect.value;

            educationOutput.innerHTML = 'Generating...';

            try {
                const response = await fetch('/api/education/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, difficulty })
                });
                const data = await response.json();

                let html = '<h3>Flashcards</h3>';
                data.flashcards.forEach(fc => {
                    html += `<p><b>Q:</b> ${fc.question}<br/><b>A:</b> ${fc.answer}</p>`;
                });

                html += '<h3>MCQs</h3>';
                data.mcqs.forEach((mcq, idx) => {
                    html += `<p><b>Q:</b> ${mcq.question}<br/>` +
                            mcq.options.map(opt => 
                                `<label><input type="radio" name="mcq-${idx}"> ${opt}</label>`).join('<br/>') + '</p>';
                });

                educationOutput.innerHTML = html;
            } catch {
                educationOutput.innerHTML = 'Error generating questions.';
            }
        });
    }

    // Admin elements
    const fileUpload = document.getElementById('file-upload');
    const uploadFileBtn = document.getElementById('upload-file-btn');
    const filesList = document.getElementById('files-list');

    async function loadFiles() {
        if (!filesList) return;
        try {
            const response = await fetch('/api/files');
            const data = await response.json();
            filesList.innerHTML = '';
            data.files.forEach(file => {
                const li = document.createElement('li');
                li.textContent = file;
                const btn = document.createElement('button');
                btn.textContent = 'Delete';
                btn.addEventListener('click', () => deleteFile(file));
                li.appendChild(btn);
                filesList.appendChild(li);
            });
        } catch {
            if (filesList) filesList.innerHTML = 'Failed to load files.';
        }
    }

    async function deleteFile(filename) {
        if (!confirm(`Delete file ${filename}?`)) return;
        try {
            const response = await fetch('/api/delete_file', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename })
            });
            const data = await response.json();
            if (data.success) {
                alert('File deleted.');
                loadFiles();
            } else {
                alert(data.error || 'Delete error.');
            }
        } catch {
            alert('Delete failed.');
        }
    }

    if (uploadFileBtn) {
        uploadFileBtn.addEventListener('click', async () => {
            if (!fileUpload || fileUpload.files.length === 0) {
                alert('Select a file to upload.');
                return;
            }
            const file = fileUpload.files[0];
            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/api/upload', { method: 'POST', body: formData });
                const data = await response.json();
                if (data.success) {
                    alert('File uploaded.');
                    fileUpload.value = '';
                    loadFiles();
                } else {
                    alert(data.error || 'Upload error.');
                }
            } catch {
                alert('Upload failed.');
            }
        });
    }

    // Load files if on admin page
    if (filesList) loadFiles();
});
