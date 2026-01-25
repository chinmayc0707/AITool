import os
import json
import shutil
import stat
import time
import gc
import itertools
import hashlib
import pickle
from typing import List, Tuple

from flask import Flask, render_template, request, jsonify, Response, session, send_from_directory
from werkzeug.utils import secure_filename
import requests
import pdfplumber
import faiss
import pytesseract
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

app = Flask(__name__)
app.secret_key = os.urandom(24)

# ───────────────────── constants ─────────────────────────────────────────
UPLOAD_DIR = "uploaded_documents"
VECTOR_DIR = "vectorstore"
SESSION_FILE = "user_session.pkl"

PROVIDERS = {
    "OpenRouter (free)": {
        "url":   "https://openrouter.ai/api/v1/chat/completions",
        "model": "google/gemma-2-9b-it:free",
        "env":   "OPENROUTER_API_KEY",
        "headers": {
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-app.example",
            "X-Title": "Multi-PDF Chat"
        },
        "stream_style": "openai"
    },
    "Local Ollama": {
        "url":   "http://localhost:11434/api/chat",
        "model": "mistral",
        "env":   None,
        "headers": {},
        "stream_style": "ollama"
    },
    "Mistral AI": {
        "url":   "https://api.mistral.ai/v1/chat/completions",
        "model": "mistral-small-latest",
        "env":   "MISTRAL_API_KEY",
        "headers": {"Content-Type": "application/json"},
        "stream_style": "openai"
    }
}

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global state to mimic the persistent session of the original app
class AppState:
    def __init__(self):
        self.api_key = ""
        self.hash2file = {}  # hash -> filename
        self.processed_hashes = set()
        self.source_files = set() # filenames
        self.vector_store = None
        self.load_session()

    def load_session(self):
        if os.path.exists(SESSION_FILE):
            try:
                data = pickle.load(open(SESSION_FILE, "rb"))
                self.api_key = data.get("api_key", "")
                self.hash2file = data.get("files", {})
            except Exception:
                pass

    def save_session(self):
        sess = {"api_key": self.api_key, "files": self.hash2file}
        pickle.dump(sess, open(SESSION_FILE, "wb"))

GLOBAL_STATE = AppState()

# ───────────────────── PDF / OCR helpers ────────────────────────────────
def display_name(fn: str) -> str:
    if "_" not in fn:
        return fn
    head, tail = fn.split("_", 1)
    ok = len(head) == 64 and all(c in "0123456789abcdefABCDEF" for c in head)
    return tail if ok else fn

def _on_rm_error(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def _safe_rmtree(path, tries=5, delay=.5):
    for _ in range(tries):
        try:
            shutil.rmtree(path, onerror=_on_rm_error)
            return
        except PermissionError:
            time.sleep(delay)
    shutil.rmtree(path, onerror=_on_rm_error)

def _sha256(b): return hashlib.sha256(b).hexdigest()

def _dedup_path(name):
    base, ext = os.path.splitext(name)
    for i in itertools.count():
        cand = f"{base}_{i}{ext}" if i else name
        full = os.path.join(UPLOAD_DIR, cand)
        if not os.path.exists(full):
            return full

def _file_type(fn): return "PDF" if fn.lower().endswith(".pdf") else "IMAGE"

def _extract_pdf(path):
    txt = ""
    try:
        with pdfplumber.open(path) as pdf:
            for pg in pdf.pages:
                if (t := pg.extract_text()):
                    txt += t + "\n"
                for tbl in pg.extract_tables():
                    for row in tbl:
                        txt += " | ".join(c or "" for c in row) + "\n"
    except Exception as e:
        print(f"Error reading {display_name(os.path.basename(path))}: {e}")
    return txt.strip()

def _extract_img(path):
    try:
        return pytesseract.image_to_string(Image.open(path)).strip()
    except Exception as e:
        print(f"OCR failed for {display_name(os.path.basename(path))}: {e}")
        return ""

def _split(text, fn, tp):
    if not text:
        return []
    head = f"File: {fn} | Type: {tp}\n"
    split = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return [head + chunk for chunk in split.split_text(text)]

def _process(path):
    fn = display_name(os.path.basename(path))
    tp = _file_type(fn)
    txt = _extract_pdf(path) if tp == "PDF" else _extract_img(path)
    return _split(txt, fn, tp), tp

# ─────────────────────── FAISS plumbing ─────────────────────────────────
# Global embeddings to mimic st.cache_resource
_EMBEDDINGS = None

def get_embeddings():
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _EMBEDDINGS

def _new_store():
    emb = get_embeddings()
    dim = len(emb.embed_query("x"))
    return FAISS(emb, faiss.IndexFlatL2(dim), InMemoryDocstore({}), {})

def load_store():
    if GLOBAL_STATE.vector_store:
        return GLOBAL_STATE.vector_store
    try:
        GLOBAL_STATE.vector_store = FAISS.load_local(VECTOR_DIR, get_embeddings())
    except Exception:
        GLOBAL_STATE.vector_store = _new_store()
    return GLOBAL_STATE.vector_store

def persist(store):
    store.save_local(VECTOR_DIR)
    GLOBAL_STATE.vector_store = store

def add_to_db(path, h):
    if h in GLOBAL_STATE.processed_hashes:
        return
    if not os.path.exists(path):
        return
    chunks, tp = _process(path)
    if not chunks:
        return
    fn = display_name(os.path.basename(path))
    store = load_store()
    store.add_texts(chunks, metadatas=[{"source": fn, "type": tp}] * len(chunks))
    persist(store)
    GLOBAL_STATE.processed_hashes.add(h)
    GLOBAL_STATE.source_files.add(fn.lower())

def _bootstrap_saved_files():
    for h, fn in list(GLOBAL_STATE.hash2file.items()):
        full = os.path.join(UPLOAD_DIR, fn)
        if os.path.exists(full):
            add_to_db(full, h)
        else:
            GLOBAL_STATE.hash2file.pop(h, None)
    GLOBAL_STATE.save_session()

_bootstrap_saved_files()

# ───────────────── retrieval helpers ────────────────────────────────────
_TYPE_KW = {"pdf": "PDF", "document": "PDF",
            "image": "IMAGE", "picture": "IMAGE", "screenshot": "IMAGE"}

def _filters(q):
    p = q.lower()
    names = [n for n in GLOBAL_STATE.source_files
             if n in p or os.path.splitext(n)[0] in p]
    types = {v for k, v in _TYPE_KW.items() if k in p}
    return names, types

def build_ctx(q, k=5):
    if not GLOBAL_STATE.processed_hashes:
        return ""
    names, types = _filters(q)
    docs = load_store().similarity_search(q, k=20)
    if names or types:
        docs = [d for d in docs if
                ((not names) or (d.metadata.get("source", "").lower() in names)) and
                ((not types) or (d.metadata.get("type") in types))]
    return "\n\n---\n\n".join(d.page_content for d in docs[:k])

# ────────────────────────── LLM streaming ───────────────────────────────
def _stream_iterator(resp, style):
    for raw in resp.iter_lines():
        if not raw or not raw.strip():
            continue
        line = raw.decode("utf-8", errors="ignore")
        if line.startswith("data:"):
            line = line[5:].lstrip()
        if line == "[DONE]":
            break
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue
        if style == "openai":
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if "content" in delta:
                yield delta["content"]
        elif style == "ollama":
            msg = chunk.get("message", {})
            if "content" in msg:
                yield msg["content"]

# ────────────────────────── Routes ──────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/files")
def list_files():
    files = []
    for fn in sorted(GLOBAL_STATE.source_files):
        files.append({"name": fn, "type": _file_type(fn)})
    return jsonify(files)

@app.route("/upload", methods=["POST"])
def upload_files():
    uploaded_files = request.files.getlist("files")
    processed = []
    for u in uploaded_files:
        if not u.filename:
            continue
        # Use secure_filename to sanitize the filename
        safe_filename = secure_filename(u.filename)
        if not safe_filename:
            continue

        data = u.read()
        h = _sha256(data)

        if h not in GLOBAL_STATE.hash2file:
            path = _dedup_path(safe_filename)
            with open(path, "wb") as f:
                f.write(data)
            GLOBAL_STATE.hash2file[h] = os.path.basename(path)

        # Add to DB
        add_to_db(os.path.join(UPLOAD_DIR, GLOBAL_STATE.hash2file[h]), h)
        processed.append(safe_filename)

    GLOBAL_STATE.save_session()
    return jsonify({"status": "ok", "processed": processed})

@app.route("/settings", methods=["POST"])
def update_settings():
    data = request.json
    provider = data.get("provider")
    api_key = data.get("api_key")

    if provider and provider in PROVIDERS:
        session["provider"] = provider

    if api_key is not None:
        GLOBAL_STATE.api_key = api_key
        GLOBAL_STATE.save_session()

    return jsonify({"status": "ok"})

@app.route("/clear", methods=["POST"])
def clear_data():
    for pth in (VECTOR_DIR, UPLOAD_DIR, SESSION_FILE):
        if os.path.isdir(pth):
            _safe_rmtree(pth)
        elif os.path.isfile(pth):
            os.remove(pth)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Reset in-memory state
    global _EMBEDDINGS
    _EMBEDDINGS = None
    GLOBAL_STATE.api_key = ""
    GLOBAL_STATE.hash2file = {}
    GLOBAL_STATE.processed_hashes = set()
    GLOBAL_STATE.source_files = set()
    GLOBAL_STATE.vector_store = None

    gc.collect()
    return jsonify({"status": "ok"})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    prompt = data.get("message", "")
    history = data.get("history", []) # List of {role: ..., content: ...}

    if not GLOBAL_STATE.processed_hashes:
        return jsonify({"error": "Upload PDFs or images first."}), 400

    ctx = build_ctx(prompt)
    if not ctx:
        pass

    full_prompt = ctx + "\n\nQ: " + prompt

    system_msg = {
        "role": "system",
        "content": ("Answer only from the context, and explain it in detail. "
                    "Strictly cite the filename(s) used, strictly include backticks around the filename. "
                    "If no answer is present, strictly say exactly: I don't know.")
    }

    llm_messages = [system_msg] + history + [{"role": "user", "content": full_prompt}]

    provider_name = session.get("provider", "Local Ollama")
    if provider_name not in PROVIDERS:
        provider_name = "Local Ollama"

    p = PROVIDERS[provider_name]
    req_data = {"model": p["model"], "messages": llm_messages, "stream": True}

    hdr = dict(p["headers"])
    if p["env"]:
        key = GLOBAL_STATE.api_key or os.getenv(p["env"])
        if not key:
            return jsonify({"error": f"Add {p['env']} in settings or env variable"}), 400
        hdr["Authorization"] = f"Bearer {key}"

    def generate():
        try:
            with requests.post(p["url"], json=req_data, headers=hdr, stream=True, timeout=90) as r:
                # Better error handling for 4xx/5xx
                if not r.ok:
                    yield f"Error {r.status_code}: {r.text}"
                    return

                # r.raise_for_status() # Already checked above
                for token in _stream_iterator(r, p["stream_style"]):
                    yield token
        except Exception as e:
            yield f"Error: {str(e)}"

    return Response(generate(), mimetype='text/plain')

if __name__ == "__main__":
    app.run(debug=True, port=5000)
