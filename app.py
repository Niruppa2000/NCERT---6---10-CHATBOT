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
# Text Cleaning
# -----------------------------
def clean_text(t):
    t = t.replace("\n", " ").replace("\x00", "")
    t = " ".join(t.split())
    return t

# -----------------------------
# PDF Loading
# -----------------------------
def load_pdfs(uploaded_files):
    texts = []
    for f in uploaded_files:
        doc = fitz.open(stream=f.read(), filetype="pdf")
        for page in doc:
            t = clean_text(page.get_text())
            if t and len(t.strip()) > 50:
                texts.append(t.strip())
    return texts

# -----------------------------
# Chunking
# -----------------------------
def chunk_text(texts, size=300):
    chunks = []
    for t in texts:
        words = t.split()
        for i in range(0, len(words), size):
            c = " ".join(words[i:i+size])
            if len(c) > 50:
                chunks.append(c)
    return chunks

# -----------------------------
# Embeddings
# -----------------------------
def build_embeddings(chunks):
    emb = embedder.encode(chunks, normalize_embeddings=True)
    return emb


# -----------------------------
# Retrieval
# -----------------------------
def retrieve_context(question, chunks, embeddings, top_k=5):
    q_emb = embedder.encode([question], normalize_embeddings=True)[0]

    sims = np.dot(embeddings, q_emb)
    ranked = sims.argsort()[::-1]

    selected = []
    for idx in ranked:
        if len(chunks[idx]) > 120:   # avoid tiny useless chunks
            selected.append(chunks[idx])
        if len(selected) == top_k:
            break

    ctx = " ".join(selected)
    return clean_text(ctx)[:1800]

# Enforce Numbered Points
# -----------------------------
def enforce_points(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    points = []

    for l in lines:
        l = re.sub(r"^[0-9]+[\.\)]\s*", "", l)
        if len(l) > 10:
            points.append(l)

    if len(points) < 5:
        sentences = re.split(r"\.\s+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        points = sentences[:5]

    while len(points) < 5:
        points.append(points[-1])

    return "\n".join([f"{i+1}. {p}." for i, p in enumerate(points[:5])])

# -----------------------------
# Answer Generation
# -----------------------------
def generate_answer(question, context):
    prompt = f"""
Using the textbook content below, answer the question.

Text:
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
            max_new_tokens=220,
            num_beams=4,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )

    raw = tokenizer.decode(out[0], skip_special_tokens=True)
    return enforce_points(raw)

# -----------------------------
# Streamlit UI
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
    if chunks is None:
        st.warning("Please upload NCERT PDFs first.")
    else:
        context = retrieve_context(question, chunks, embeddings)
        answer = generate_answer(question, context)
        st.markdown("### Answer")
        st.markdown(answer)

