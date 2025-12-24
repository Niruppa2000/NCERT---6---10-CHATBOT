import streamlit as st
import torch
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="NCERT Chatbot", layout="centered")
st.title("📘 NCERT Class 6–10 Chatbot")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)
    model.eval()
    return embedder, tokenizer, model

embedder, tokenizer, model = load_models()

def load_pdfs(uploaded_files):
    texts = []
    for f in uploaded_files:
        doc = fitz.open(stream=f.read(), filetype="pdf")
        for page in doc:
            t = page.get_text()
            if t and len(t.strip()) > 50:
                texts.append(t.strip())
    return texts

def chunk_text(texts, size=300):
    chunks = []
    for t in texts:
        words = t.split()
        for i in range(0, len(words), size):
            c = " ".join(words[i:i+size])
            if len(c) > 50:
                chunks.append(c)
    return chunks

def build_embeddings(chunks):
    emb = embedder.encode(chunks)
    return emb

def retrieve_context(question, chunks, embeddings, top_k=2):
    q_emb = embedder.encode([question])[0]
    sims = np.dot(embeddings, q_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb))
    top_idx = sims.argsort()[-top_k:][::-1]
    return " ".join([chunks[i] for i in top_idx])

def generate_answer(question, context):
    prompt = f"""
Use the following textbook content to answer the question.

Textbook content:
{context}

Question:
{question}

Answer in 5 short numbered points:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            num_beams=4,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)

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
