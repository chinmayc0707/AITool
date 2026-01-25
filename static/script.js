document.addEventListener('DOMContentLoaded', () => {
    loadFiles();
    setupEventListeners();
});

let chatHistory = [];

function setupEventListeners() {
    // Settings
    document.getElementById('provider-select').addEventListener('change', updateSettings);
    document.getElementById('save-settings-btn').addEventListener('click', updateSettings);

    // Clear Data
    document.getElementById('clear-data-btn').addEventListener('click', clearData);

    // Chat
    document.getElementById('chat-form').addEventListener('submit', handleChatSubmit);
}

async function loadFiles() {
    try {
        const response = await fetch('/files');
        const files = await response.json();
        const list = document.getElementById('file-list');
        list.innerHTML = '';
        files.forEach(f => {
            const li = document.createElement('li');
            li.textContent = `• ${f.name} (${f.type})`;
            list.appendChild(li);
        });
    } catch (error) {
        console.error('Error loading files:', error);
    }
}

async function uploadFiles(inputId) {
    const input = document.getElementById(inputId);
    const files = input.files;
    if (files.length === 0) return;

    const spinner = document.getElementById('loading-spinner');
    spinner.style.display = 'block';

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        if (result.status === 'ok') {
            loadFiles();
            input.value = ''; // clear input
        } else {
            alert('Upload failed');
        }
    } catch (error) {
        console.error('Error uploading:', error);
        alert('Error uploading files');
    } finally {
        spinner.style.display = 'none';
    }
}

async function updateSettings() {
    const provider = document.getElementById('provider-select').value;
    const apiKey = document.getElementById('api-key').value;

    const data = { provider };
    if (apiKey) data.api_key = apiKey;

    try {
        await fetch('/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        // alert('Settings updated');
    } catch (error) {
        console.error('Error updating settings:', error);
    }
}

async function clearData() {
    if (!confirm('Are you sure you want to clear all data? This cannot be undone.')) return;

    try {
        await fetch('/clear', { method: 'POST' });
        location.reload();
    } catch (error) {
        console.error('Error clearing data:', error);
    }
}

async function handleChatSubmit(e) {
    e.preventDefault();
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (!message) return;

    // Add user message
    addMessage('user', message);
    chatHistory.push({ role: 'user', content: message });
    input.value = '';

    // Create assistant message placeholder
    const assistantMsgContent = addMessage('assistant', '');
    let fullResponse = "";

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                history: chatHistory
            })
        });

        if (!response.ok) {
            const err = await response.json();
            assistantMsgContent.textContent = "Error: " + (err.error || "Unknown error");
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value, { stream: true });
            fullResponse += chunk;
            assistantMsgContent.innerHTML = DOMPurify.sanitize(marked.parse(fullResponse));
            // Scroll to bottom
            const container = document.getElementById('messages-container');
            container.scrollTop = container.scrollHeight;
        }

        chatHistory.push({ role: 'assistant', content: fullResponse });

    } catch (error) {
        console.error('Chat error:', error);
        assistantMsgContent.textContent += "\n[Error generating response]";
    }
}

function addMessage(role, text) {
    const container = document.getElementById('messages-container');
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}-message`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = text;

    msgDiv.appendChild(contentDiv);
    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;

    return contentDiv;
}
