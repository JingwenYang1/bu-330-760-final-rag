import streamlit as st
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import google.genai as genai
from google.genai import types

@st.cache_resource
def load_resources():
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    return chunks, embeddings, embed_model, bm25

chunks, embeddings, embed_model, bm25 = load_resources()
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

def hybrid_search(query, top_k=10):
    query_emb = embed_model.encode([query])
    sem_scores = np.dot(embeddings, query_emb.T).flatten()
    sem_scores = sem_scores / (sem_scores.max() + 1e-9)
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_scores = bm25_scores / (bm25_scores.max() + 1e-9)
    combined = 0.5 * sem_scores + 0.5 * bm25_scores
    top_idx = combined.argsort()[-top_k:][::-1]
    return [(chunks[i], combined[i]) for i in top_idx]

def ask_llm(query, retrieved):
    parts = []
    for c, s in retrieved:
        header = c.get("header", "")
        pages = c.get("pages", [0])
        page_str = ", ".join(str(p) for p in pages)
        parts.append(f"[Section: {header}, Pages: {page_str}]\n{c['text']}")
    context = "\n\n---\n\n".join(parts)
    prompt = (
        "You are a D&D 5e rules assistant. Answer the rule question based ONLY on the SRD sections provided below.\n\n"
        "Your response must follow this exact format:\n\n"
        "## Ruling\n"
        "State the conclusion directly in 1-2 sentences.\n\n"
        "## Explanation\n"
        "Explain the reasoning step by step. When referencing a rule, mention the page number, "
        "for example: 'On page 12, in the section on Grappling, the SRD states that...' "
        "or 'According to page 45, the Combat rules regarding bonus actions...'. "
        "Do NOT use direct quotes in this section.\n\n"
        "## SRD References\n"
        "For each rule you relied on, list:\n"
        "- A short descriptive title and page number (e.g. 'Two-Weapon Fighting (p. 42)')\n"
        "- The exact quote from the SRD\n\n"
        "Based on the retrieved rules, always provide your best ruling and reasoning. "
        "Only say the rules are unclear if the retrieved text contains NO relevant information at all.\n\n"
        "RETRIEVED SRD SECTIONS:\n" + context + "\n\n"
        "QUESTION: " + query
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0)
    )
    return response.text

st.title("D&D 5e Rules Assistant")
st.caption("Ask any rule question — answers are based on the official SRD 5.2 (hybrid semantic + keyword search)")

query = st.text_input("Ask a rule question:", placeholder="e.g. Can a wizard cast a spell while holding a shield and weapon?")

if st.button("Submit") and query:
    with st.spinner("Retrieving and reasoning..."):
        results = hybrid_search(query)
        answer = ask_llm(query, results)
    st.session_state["last_answer"] = answer
    st.session_state["last_results"] = results

if "last_answer" in st.session_state:
    st.markdown(st.session_state["last_answer"])
    with st.expander("Developer View: Retrieved SRD Chunks"):
        for c, score in st.session_state["last_results"]:
            pages = c.get("pages", [0])
            page_str = ", ".join(str(p) for p in pages)
            st.markdown(f"**{c['header']}** (p. {page_str}) — score: {score:.3f}")
            st.text(c["text"][:300] + "...")
            st.divider()
