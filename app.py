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
        "It promotes equality and justice
