import streamlit as st
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import google.genai as genai

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

def rag_search(query, top_k=5, threshold=0.2):
    query_emb = embed_model.encode([query])
    scores = np.dot(embeddings, query_emb.T).flatten()
    top_idx = scores.argsort()[-top_k:][::-1]
    return [(chunks[i], scores[i]) for i in top_idx if scores[i] >= threshold]

def bm25_search(query, top_k=5):
    scores = bm25.get_scores(query.lower().split())
    top_idx = scores.argsort()[-top_k:][::-1]
    return [(chunks[i], scores[i]) for i in top_idx]

def ask_llm(query, retrieved):
    context = "\n\n---\n\n".join(
        [f"[Section: {c['header']}]\n{c['text']}" for c, s in retrieved]
    )
    prompt = f"""You are a D&D 5e rules assistant. Based ONLY on the following SRD sections, answer the rule question. Cite which sections you used. If the answer is not clearly supported by the provided text, say so.

RETRIEVED RULES:
{context}

QUESTION: {query}

RULING:"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

st.title("D&D 5e Rules Assistant")
st.caption("Ask any rule question and get an answer based on the official SRD 5.2")

query = st.text_input("Ask a rule question:", placeholder="e.g. Can a wizard cast a spell while holding a shield and weapon?")

if st.button("Submit") and query:
    st.session_state["query"] = query
    with st.spinner("Retrieving and reasoning..."):
        st.session_state["rag_results"] = rag_search(query)
        st.session_state["answer"] = ask_llm(query, st.session_state["rag_results"])
        st.session_state["bm25_results"] = bm25_search(query)

if "answer" in st.session_state:
    st.subheader("Answer")
    st.markdown(st.session_state["answer"])

    st.subheader("Referenced SRD Sections")
    for chunk, score in st.session_state["rag_results"]:
        with st.expander(chunk['header'] if chunk['header'] else "SRD Section"):
            st.markdown(chunk['text'])

    with st.expander("Show keyword search baseline comparison"):
        st.subheader("BM25 Keyword Search Results")
        for chunk, score in st.session_state["bm25_results"]:
            with st.expander(chunk['header'] if chunk['header'] else "SRD Section"):
                st.markdown(chunk['text'])
