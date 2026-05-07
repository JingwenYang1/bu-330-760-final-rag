import streamlit as st
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import google.genai as genai

@st.cache_resource
def load_resources():
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return chunks, embeddings, embed_model

chunks, embeddings, embed_model = load_resources()
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

def rag_search(query, top_k=5):
    query_emb = embed_model.encode([query])
    scores = np.dot(embeddings, query_emb.T).flatten()
    top_idx = scores.argsort()[-top_k:][::-1]
    return [(chunks[i], scores[i]) for i in top_idx]

def ask_llm(query, retrieved):
    context = "\n\n---\n\n".join(
        [f"[Section: {c.get('header','')}, Page: {c.get('pages',[0])[0]}]\n{c['text']}" for c, s in retrieved]
    )
    prompt = f"""You are a D&D 5e rules assistant. Answer the rule question based ONLY on the SRD sections provided below.

Your response must follow this exact format:

## Ruling
State the conclusion directly in 1-2 sentences.

## Explanation
Explain the reasoning step by step. When referencing a rule, mention the page number, for example: "On page 12, in the section on Grappling, the SRD states that..." or "According to page 45, the Combat rules regarding bonus actions...". Do NOT use direct quotes in this section.

## SRD References
For each rule you relied on, list:
- A short descriptive title and page number (e.g. "Two-Weapon Fighting (p. 42)")
- The exact quote from the SRD

If the answer is not clearly supported by the provided text, say so explicitly.

RETRIEVED SRD SECTIONS:
{context
