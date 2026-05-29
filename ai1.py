# ───────────────────────── Chat with Multiple PDFs & Images ──────────────
# perpl.py  –  persistent keys + files   2025-08-22

import os, json, shutil, stat, time, gc, itertools, hashlib, pickle
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import streamlit as st
import requests, pdfplumber, faiss, pytesseract
from PIL import Image
Image.MAX_IMAGE_PIXELS = 100_000_000
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

# ───────────────────── constants ─────────────────────────────────────────
UPLOAD_DIR = "uploaded_documents"
VECTOR_DIR = "vectorstore"

PROVIDERS = {
    "OpenRouter (free)": {
        "url":   "https://openrouter.ai/api/v1/chat/completions",
        "model": "mistralai/mistral-7b-instruct:free",
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

@st.cache_data(ttl=3600)
def get_openrouter_models():
    try:
        r = requests.get("https://openrouter.ai/api/v1/models", timeout=5)
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]
    except Exception:
        return ["mistralai/mistral-7b-instruct:free", "openai/gpt-3.5-turbo", "meta-llama/llama-3-8b-instruct"]

@st.cache_data(ttl=3600)
def get_ollama_models(host):
    try:
        r = requests.get(f"{host}/api/tags", timeout=2)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return ["mistral", "llama3"]

# ────────── helper: strip legacy hash prefix ───────────────
def display_name(fn: str) -> str:
    if "_" not in fn:
        return fn
    head, tail = fn.split("_", 1)
    ok = len(head) == 64 and all(c in "0123456789abcdefABCDEF" for c in head)
    return tail if ok else fn

# ───────────── helpers: Windows-safe rmtree ──────────────────────────────
def _sha256(b): return hashlib.sha256(b).hexdigest()

def init_session():
    defaults = dict(
        messages=[{"role": "assistant",
                   "content": "Upload PDFs or images and ask me anything about them!"}],
        processed_hashes=set(),
        processed_uploads=set(),
        source_files=set(),
        uploader_key=0,
        provider="Local Ollama",
        ollama_host="http://localhost:11434",
        ollama_model="mistral",
        openrouter_model="mistralai/mistral-7b-instruct:free",
        api_key="",
        hash2file={},
        vector_store=None
    )
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

# ───────────────────── PDF / OCR helpers ────────────────────────────────
def _file_type(fn): return "PDF" if fn.lower().endswith(".pdf") else "IMAGE"

def _extract_pdf(fobj, fn):
    txt_parts = []
    try:
        with pdfplumber.open(fobj) as pdf:
            for pg in pdf.pages:
                if (t := pg.extract_text()):
                    txt_parts.append(t + "\n")
                for tbl in pg.extract_tables():
                    for row in tbl:
                        txt_parts.append(" | ".join(c or "" for c in row) + "\n")
    except Exception as e:
        st.error(f"Error reading {display_name(fn)}: {e}")
    return "".join(txt_parts).strip()

def _extract_img(fobj, fn):
    try:
        return pytesseract.image_to_string(Image.open(fobj)).strip()
    except Exception as e:
        st.error(f"OCR failed for {display_name(fn)}: {e}")
        return ""

@st.cache_resource
def _get_splitter():
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def _split(text, fn, tp):
    if not text:
        return []
    head = f"File: {fn} | Type: {tp}\n"
    split = _get_splitter()
    return [head + chunk for chunk in split.split_text(text)]

def _process(fobj, fn):
    fn_display = display_name(fn)
    tp = _file_type(fn_display)
    txt = _extract_pdf(fobj, fn_display) if tp == "PDF" else _extract_img(fobj, fn_display)
    return _split(txt, fn_display, tp), tp

# ─────────────────────── FAISS plumbing ─────────────────────────────────
@st.cache_resource
def _emb(): 
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cuda" if faiss.get_num_gpus() > 0 else "cpu"})

def _new_store():
    dim = len(_emb().embed_query("x"))
    st.session_state["vector_store"] = FAISS(_emb(), faiss.IndexFlatL2(dim), InMemoryDocstore({}), {})
    return st.session_state["vector_store"]

def load_store():
    if st.session_state.get("vector_store") is not None:
        return st.session_state["vector_store"]
    return _new_store()

def persist(store): st.session_state["vector_store"] = store

# ───────────────── ingestion ────────────────────────────────────────────
def add_many_to_db(files):
    if not files:
        return

    all_chunks = []
    all_metadatas = []
    seen_in_batch = set()
    added_names = set()
    store = load_store()

    for u in files:
        # Stream chunks to compute hash
        hasher = hashlib.sha256()
        u.seek(0)
        while True:
            chunk = u.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
        h = hasher.hexdigest()
        u.seek(0)

        uid = f"{u.name}-{u.size}"
        if uid in st.session_state.processed_uploads:
            continue

        if h in st.session_state.processed_hashes or h in seen_in_batch:
            st.session_state.processed_uploads.add(uid)
            continue

        seen_in_batch.add(h)

        with st.spinner(f"Processing {u.name} …"):
            chunks, tp = _process(u, u.name)
            if not chunks:
                st.warning(f"Skipped empty/unsupported: {display_name(u.name)}")
            else:
                fn = display_name(u.name)
                all_chunks.extend(chunks)
                all_metadatas.extend([{"source": fn, "type": tp}] * len(chunks))
                added_names.add(fn.lower())
                st.success(f"Added: {u.name}")

            st.session_state.processed_hashes.add(h)
            st.session_state.processed_uploads.add(uid)

    if all_chunks:
        store.add_texts(all_chunks, metadatas=all_metadatas)
        persist(store)
        st.session_state.source_files.update(added_names)



def _handle(files):
    if files:
        add_many_to_db(files)

def sidebar():
    with st.sidebar:
        st.header("⚙ Controls")
        st.session_state.provider = st.selectbox(
            "LLM provider",
            list(PROVIDERS),
            index=list(PROVIDERS).index(st.session_state.provider)
        )

        provider_name = st.session_state.provider
        
        if provider_name == "Local Ollama":
            st.session_state.ollama_host = st.text_input("Ollama Host URL", value=st.session_state.ollama_host)
            models = get_ollama_models(st.session_state.ollama_host.rstrip('/'))
            curr_model = st.session_state.ollama_model
            if curr_model not in models:
                models.append(curr_model)
            st.session_state.ollama_model = st.selectbox("Ollama Model", models, index=models.index(curr_model))
        
        elif provider_name == "OpenRouter (free)":
            models = get_openrouter_models()
            curr_model = st.session_state.openrouter_model
            if curr_model not in models:
                models.append(curr_model)
            st.session_state.openrouter_model = st.selectbox("OpenRouter Model", models, index=models.index(curr_model))

        env = PROVIDERS[provider_name]["env"]
        if env:
            k = st.text_input(
                "API key",
                type="password",
                value=st.session_state.api_key,
                placeholder=env
            )
            if k != st.session_state.api_key:
                st.session_state.api_key = k

        _handle(st.file_uploader(
            "Upload PDFs", type="pdf", accept_multiple_files=True,
            key=f"pdf{st.session_state.uploader_key}"
        ))
        _handle(st.file_uploader(
            "Upload Images", type=["png", "jpg", "jpeg", "bmp", "tiff"],
            accept_multiple_files=True,
            key=f"img{st.session_state.uploader_key}"
        ))

        if st.button("Clear all data"):
            st.cache_resource.clear()
            st.session_state.clear()
            gc.collect()
            st.success("All data cleared – reloading…")
            st.rerun()

def show_files():
    if not st.session_state.source_files:
        return
    st.subheader("📂 Uploaded files")
    for fn in sorted(st.session_state.source_files):
        st.caption(f"• {fn} ({_file_type(fn)})")

# ─────────────────────────── main ───────────────────────────────────────
def main():
    st.set_page_config(page_title="Multi-Document Chat", layout="wide")
    st.title("Chat with Multiple PDFs & Images")

    init_session()
    sidebar()
    show_files()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Ask a question…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)
        spot = st.empty()
        if not st.session_state.processed_hashes:
            spot.warning("Upload PDFs or images first.")
            return
        ctx = build_ctx(prompt)
        if not ctx:
            spot.info("No relevant context found.")
        ans = chat_llm(ctx + "\n\nQ: " + prompt, spot)
        st.session_state.messages.append({"role": "assistant", "content": ans})

if __name__ == "__main__":
    main()
