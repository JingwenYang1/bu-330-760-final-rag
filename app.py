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
    parts = []
    for c, s in retrieved:
        header = c.get("header", "")
        page = c.get("pages", [0])[0]
        parts.append("[Section: " + header + ", Page: " + str(page) + "]\n" + c["text"])
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
        "If the answer is not clearly supported by the provided text, say so explicitly.\n\n"
        "RETRIEVED SRD SECTIONS:\n" + context + "\n\n"
        "QUESTION: " + query
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"temperature": 0}
    )
    return response.text

st.title("D&D 5e Rules Assistant")
st.caption("Ask any rule question and get an answer based on the official SRD 5.2")

query = st.text_input("Ask a rule question:", placeholder="e.g. Can a wizard cast a spell while holding a shield and weapon?")

if st.button("Submit") and query:
    with st.spinner("Retrieving and reasoning..."):
        results = rag_search(query)
        answer = ask_llm(query, results)
    st.markdown(answer)
