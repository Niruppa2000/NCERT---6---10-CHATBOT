import streamlit as st
import torch
import fitz
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="NCERT Chatbot", layout="centered")
st.title("📘 NCERT Class 6–10 Chatbot")

# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Load Models (cached)
# -----------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)
    model.eval()
    return embedder, tokenizer, model

embedder, tokenizer, model = load_models()

# -----------------------
# PDF Processing
# -----------------------
def load_pdfs(uploaded_files):
    texts = []
    for f in uploaded_files:
        doc = fitz.open(stream=f.read(), filetype="pdf")
        for page in doc:
            t = page.get_text()
            if t and len(t.strip()) > 50:
                texts.append(t.strip())
    return texts

def chunk_text(texts, size=350):
    chunks = []
    for t in texts:
        words = t.split()
        for i in range(0, len(words), size):
            c = " ".join(words[i:i+size])
            if len(c) > 50:
                chunks.append(c)
    return chunks

# -----------------------
# Vector Store
# -----------------------
def build_index(chunks):
    emb = embedder.encode(chunks)
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(np.array(emb))
    return index

# -----------------------
# Generate Aspects
# -----------------------
def generate_aspects(question):
    aspects = {
        "Definition": f"In school textbooks, what is {question}?",
        "How": f"How does {question} work or what does it do?",
        "Importance": f"Why is {question} important?",
        "Example": f"Give one simple example or effect of {question}.",
        "Conclusion": f"Summarize {question} in one line."
    }

    results = {}

    for key, q in aspects.items():
        prompt = f"Answer only using academic knowledge.\n{q}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80, num_beams=3)

        results[key] = tokenizer.decode(out[0], skip_special_tokens=True).strip()

    return results

# -----------------------
# Filter & Format
# -----------------------
def is_valid_point(text):
    bad = ["song", "album", "movie", "celebrity", "released", "actor", "singer"]
    return not any(b in text.lower() for b in bad)

def semantic_dedupe(points):
    unique = []
    for p in points:
        if all(p.lower() not in u.lower() for u in unique):
            unique.append(p)
    return unique

def aspects_to_points(results):
    points = []
    for k in ["Definition", "How", "Importance", "Example", "Conclusion"]:
        t = results.get(k, "").strip()
        if t and is_valid_point(t):
            points.append(t)

    unique = semantic_dedupe(points)

    while len(unique) < 5:
        unique.append(unique[-1])

    return "\n".join([f"{i+1}. {p}." for i, p in enumerate(unique[:5])])

# -----------------------
# Main Generator
# -----------------------
def generate_answer(question):
    results = generate_aspects(question)
    return aspects_to_points(results)

# -----------------------
# Streamlit UI
# -----------------------
st.sidebar.header("Upload NCERT PDFs")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    texts = load_pdfs(uploaded_files)
    chunks = chunk_text(texts)
    index = build_index(chunks)
    st.sidebar.success(f"Loaded {len(chunks)} text chunks.")

question = st.text_input("Ask a question:")

if st.button("Get Answer") and question:
    answer = generate_answer(question)
    st.markdown("### Answer")
    st.markdown(answer)
