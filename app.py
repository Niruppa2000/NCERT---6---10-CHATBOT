import streamlit as st
import torch
import fitz
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="NCERT Chatbot", layout="centered")
st.title("📘 NCERT Class 6–10 Chatbot")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Models ----------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)
    model.eval()
    return embedder, tokenizer, model

embedder, tokenizer, model = load_models()

# ---------------- Fallback Answers ----------------
FALLBACK_ANSWERS = {
    "fibre to fabric": [
        "Fibre is the raw material used to make fabric.",
        "Fibres can be natural like cotton and wool or synthetic like nylon.",
        "Fibres are spun into yarn through a process called spinning.",
        "Yarn is woven or knitted to make fabric.",
        "Fabric is used to make clothes and many other useful items."
    ],
    "photosynthesis": [
        "Photosynthesis is the process by which green plants make their own food.",
        "It uses sunlight, carbon dioxide and water to produce glucose.",
        "Chlorophyll helps in absorbing sunlight.",
        "Oxygen is released as a by-product of photosynthesis.",
        "This process is essential for plant growth and life on Earth."
    ],
    "democracy": [
        "Democracy is a form of government in which people choose their leaders.",
        "Citizens participate in decision making through elections.",
        "The government is accountable to the people.",
        "Democracy protects the rights and freedoms of citizens.",
        "It promotes equality and justice in society."
    ]
}

# ---------------- Helpers ----------------
def clean_text(t):
    t = t.replace("\n", " ").replace("\x00", "")
    return " ".join(t.split())

def load_pdfs(uploaded_files):
    texts = []
    for f in uploaded_files:
        doc = fitz.open(stream=f.read(), filetype="pdf")
        for page in doc:
            t = clean_text(page.get_text())
            if len(t) > 50:
                texts.append(t)
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
    return embedder.encode(chunks, normalize_embeddings=True)

def extract_keywords(question):
    stop = {"what","is","the","about","explain","process","of","and","to","in"}
    words = re.findall(r"\w+", question.lower())
    return [w for w in words if w not in stop and len(w) > 3]

def retrieve_context(question, chunks, embeddings, top_k=5):
    keywords = extract_keywords(question)
    filtered = [c for c in chunks if any(k in c.lower() for k in keywords)]
    use_chunks = filtered if len(filtered) >= 3 else chunks
    use_embeddings = embedder.encode(use_chunks, normalize_embeddings=True)

    q_emb = embedder.encode([question], normalize_embeddings=True)[0]
    sims = np.dot(use_embeddings, q_emb)
    ranked = sims.argsort()[::-1]

    selected = []
    for idx in ranked:
        if len(use_chunks[idx]) > 120:
            selected.append(use_chunks[idx])
        if len(selected) == top_k:
            break

    return clean_text(" ".join(selected))[:1800]

def enforce_points(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    points = []

    for l in lines:
        l = re.sub(r"^[0-9]+[\.\)]\s*", "", l)
        if len(l) > 10:
            points.append(l)

    if len(points) < 5:
        sentences = re.split(r"\.\s+", text)
        points = [s.strip() for s in sentences if len(s.strip()) > 10][:5]

    if not points:
        points = ["Information not found clearly in the textbook"]

    while len(points) < 5:
        points.append(points[-1])

    return "\n".join([f"{i+1}. {p}." for i,p in enumerate(points[:5])])

# ---------------- Answer ----------------
def generate_answer(question, context):
    q = question.lower()

    for key in FALLBACK_ANSWERS:
        if key in q:
            return "\n".join([f"{i+1}. {p}." for i,p in enumerate(FALLBACK_ANSWERS[key])])

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
    inputs = {k: v.to(device) for k,v in inputs.items()}

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

# ---------------- UI ----------------
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


