import streamlit as st
import os, re
import numpy as np
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# -----------------------
# Config
# -----------------------
MODEL_NAME = "google/flan-t5-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Load models
# -----------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    return embedder, tokenizer, model

embedder, tokenizer, model = load_models()

# -----------------------
# Load PDFs
# -----------------------
def load_pdfs(files):
    texts = []
    for file in files:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page in doc:
            t = page.get_text()
            if len(t.strip()) > 100:
                texts.append(t)
    return texts

# -----------------------
# Build embeddings
# -----------------------
def embed_texts(texts):
    return embedder.encode(texts, convert_to_numpy=True)

# -----------------------
# Retrieve context using cosine similarity
# -----------------------
def retrieve_context(question, texts, embeddings, k=5):
    q_emb = embedder.encode([question], convert_to_numpy=True)[0]
    sims = embeddings @ q_emb / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-8)
    top_idx = np.argsort(sims)[-k:][::-1]

    chunks = [texts[i] for i in top_idx]

    # keyword overlap filter
    q_words = set(re.findall(r"\w+", question.lower()))
    filtered = []
    for c in chunks:
        if len(q_words.intersection(set(re.findall(r"\w+", c.lower())))) >= 2:
            filtered.append(c)

    return " ".join(filtered)[:3000]

# -----------------------
# Enforce 5 points, no duplicates
# -----------------------
def enforce_points(text):
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"\.\s+", text)
    clean = []

    for s in sentences:
        s = s.strip()
        if len(s) > 25 and s not in clean:
            clean.append(s)

    if not clean:
        return "⚠ Could not find relevant information in the textbook for this question."

    clean = clean[:5]
    while len(clean) < 5:
        clean.append(clean[-1])

    return "\n".join([f"{i+1}. {c}." for i, c in enumerate(clean)])

# -----------------------
# Generate Answer
# -----------------------
def generate_answer(question, context):
    prompt = f"""
Answer the question ONLY using the textbook content below.
Write exactly 5 clear, simple numbered points.
Do not repeat sentences.
Do not give introductions.
Do not repeat the question.

Textbook:
{context}

Question:
{question}
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=256)
    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return enforce_points(raw)

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="NCERT Class 6–10 Chatbot", layout="wide")
st.title("📘 NCERT Class 6–10 Chatbot")

uploaded_files = st.sidebar.file_uploader("Upload NCERT PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    texts = load_pdfs(uploaded_files)
    embeddings = embed_texts(texts)
    st.sidebar.success(f"Loaded {len(texts)} chunks from PDFs.")

question = st.text_input("Ask a question:")

if st.button("Get Answer") and uploaded_files and question:
    context = retrieve_context(question, texts, embeddings)
    answer = generate_answer(question, context)
    st.markdown("### Answer")
    st.text(answer)
