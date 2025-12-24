import streamlit as st
import torch
import fitz
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="NCERT Chatbot", layout="centered")
st.title("📘 NCERT Class 6–10 Chatbot")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)
    model.eval()
    return embedder, tokenizer, model

embedder, tokenizer, model = load_models()

# -----------------------------
# Clean text
# -----------------------------
def clean_text(t):
    t = t.replace("\n", " ").replace("\x00", "")
    t = re.sub(r"\s+", " ", t)
    return t.strip()

# -----------------------------
# Load PDFs
# -----------------------------
def load_pdfs(uploaded_files):
    texts = []
    for f in uploaded_files:
        doc = fitz.open(stream=f.read(), filetype="pdf")
        for page in doc:
            t = clean_text(page.get_text())
            if len(t) > 80:
                texts.append(t)
    return texts

# -----------------------------
# Chunking
# -----------------------------
def chunk_text(texts, size=280):
    chunks = []
    for t in texts:
        words = t.split()
        for i in range(0, len(words), size):
            c = " ".join(words[i:i+size])
            if len(c) > 120:
                chunks.append(c)
    return chunks

# -----------------------------
# Embeddings
# -----------------------------
def build_embeddings(chunks):
    return embedder.encode(chunks, normalize_embeddings=True)

# -----------------------------
# Keyword extraction
# -----------------------------
def extract_keywords(question):
    stop = {"what", "is", "the", "about", "explain", "process", "of", "and", "to", "in"}
    words = re.findall(r"\w+", question.lower())
    return [w for w in words if w not in stop and len(w) > 3]

# -----------------------------
# Retrieval
# -----------------------------
def retrieve_context(question, chunks, embeddings, top_k=6):
    keywords = extract_keywords(question)

    # Prefer keyword-matching chunks
    filtered = [c for c in chunks if any(k in c.lower() for k in keywords)]
    use_chunks = filtered if len(filtered) >= 3 else chunks

    use_embeddings = embedder.encode(use_chunks, normalize_embeddings=True)
    q_emb = embedder.encode([question], normalize_embeddings=True)[0]

    sims = np.dot(use_embeddings, q_emb)
    ranked = sims.argsort()[::-1]

    selected = []
    for idx in ranked:
        selected.append(use_chunks[idx])
        if len(selected) == top_k:
            break

    ctx = " ".join(selected)
    return clean_text(ctx)[:1800]

# -----------------------------
# Enforce numbered points
# -----------------------------
def enforce_points(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    points = []

    for l in lines:
        l = re.sub(r"^[0-9]+[\.\)]\s*", "", l)
        if len(l) > 20:
            points.append(l)

    if len(points) < 5:
        sentences = re.split(r"\.\s+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        points = sentences[:5]

    if not points:
        return "⚠ Could not generate a clear answer from the uploaded textbook."

    while len(points) < 5:
        points.append(points[-1])

    return "\n".join([f"{i+1}. {p}." for i, p in enumerate(points[:5])])

# -----------------------------
# Generate answer
# -----------------------------
def generate_answer(question, context):
    prompt = f"""
Answer ONLY from the textbook content below.

Textbook content:
{context}

Question:
{question}

Write exactly 5 short numbered points:
1.
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=240,
            num_beams=4,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3
        )

    raw = tokenizer.decode(out[0], skip_special_tokens=True)
    return enforce_points(raw)

# -----------------------------
# UI
# -----------------------------
st.sidebar.header("Upload NCERT PDFs")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    texts = load_pdfs(uploaded_files)
    chunks = chunk_text(texts)
    embeddings = build_embeddings(chunks)
    st.sidebar.success(f"Loaded {len(chunks)} chunks from PDFs.")
else:
    chunks, embeddings = None, None

question = st.text_input("Ask a question:")

if st.button("Get Answer") and question:
    if not chunks:
        st.warning("Please upload NCERT PDFs first.")
    else:
        context = retrieve_context(question, chunks, embeddings)
        answer = generate_answer(question, context)
        st.markdown("### Answer")
        st.markdown(answer)
