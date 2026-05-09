
import streamlit as st
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import google.genai as genai
from google.genai import types

# --- Constants ---
SCORE_THRESHOLD = 0.10  # Context dilution fix (M3 slide 60): drop low-relevance chunks
MAX_CHUNKS = 10

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

def hybrid_search(query, top_k=MAX_CHUNKS):
    """Hybrid retrieval: 50% semantic + 50% BM25, with score threshold filtering."""
    query_emb = embed_model.encode([query])
    sem_scores = np.dot(embeddings, query_emb.T).flatten()
    sem_scores = sem_scores / (sem_scores.max() + 1e-9)
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_scores = bm25_scores / (bm25_scores.max() + 1e-9)
    combined = 0.5 * sem_scores + 0.5 * bm25_scores
    top_idx = combined.argsort()[-top_k:][::-1]
    # Context dilution fix: filter out chunks below score threshold
    results = [(chunks[i], combined[i]) for i in top_idx if combined[i] >= SCORE_THRESHOLD]
    if not results:
        # Fallback: always return at least top 3 even if below threshold
        top_idx_fallback = combined.argsort()[-3:][::-1]
        results = [(chunks[i], combined[i]) for i in top_idx_fallback]
    return results

def reorder_for_attention(retrieved):
    """Lost-in-the-middle mitigation (M3 slide 58):
    Place highest-scored chunks at beginning and end of context,
    lower-scored chunks in the middle, to match LLM attention patterns."""
    if len(retrieved) <= 2:
        return retrieved
    sorted_by_score = sorted(retrieved, key=lambda x: x[1], reverse=True)
    reordered = []
    for i, item in enumerate(sorted_by_score):
        if i % 2 == 0:
            reordered.insert(0, item)  # high scores go to front
        else:
            reordered.append(item)     # next-highest go to end
    return reordered

# --- Prompt with course concepts applied ---
SYSTEM_PROMPT = """## Role
You are a D&D 5e rules assistant. You answer rule questions based ONLY on the retrieved SRD 5.2 sections provided below.

## Task
Given a player's rule question and retrieved SRD text, provide an accurate ruling with page-number citations.

Before giving your ruling, first identify which specific rules from the retrieved text apply to this situation, and check whether ALL conditions of each rule are met (chain-of-thought, M3 slides 42-44).

## Constraints
- Base your answer ONLY on the provided SRD sections. Do not use outside knowledge.
- When analyzing a situation, carefully check whether ALL requirements and conditions of a rule are met before concluding that the rule applies or does not apply.
- When a question involves multiple game mechanics, analyze each mechanic separately before combining them into a final ruling.
- When determining if a character can perform an action, trace the full logical chain: identify what the action requires, check what the character's hands/body/resources are doing, and determine if each requirement is satisfied or blocked.
- Always provide your best ruling and reasoning. Only say the rules are unclear if the retrieved text contains NO relevant information at all. the retrieved text contains NO relevant information at all.

## Output Format
Respond in this exact format:

## Ruling
State the conclusion directly in 1-2 sentences.

## Explanation
Explain the reasoning step by step. When referencing a rule, mention the page number, for example: 'On page 12, the SRD states that...'

## SRD References
For each rule you relied on, list:
- A short descriptive title and page number (e.g. 'Two-Weapon Fighting (p. 42)')
- The exact quote from the SRD

## Confidence
State one of: **High** (SRD text directly answers the question), **Medium** (answer requires interpretation or synthesis of multiple rules), or **Low** (SRD text only partially covers this topic).
If any aspect is not covered by the retrieved text, briefly note what is missing.

## Few-Shot Example

QUESTION: Can a character use a bonus action before their action on their turn?
ANSWER:
## Ruling
Yes, a character can use their bonus action at any point during their turn, including before their action.

## Explanation
On page 15, the SRD describes the structure of a turn in combat. A turn consists of movement and an action, with the possibility of a bonus action if a feature grants one. The rules do not mandate a specific order for action and bonus action within a turn.

## SRD References
- **Turn Structure (p. 15)**: "On your turn, you can move a distance up to your Speed and take one action."
- **Bonus Actions (p. 15)**: "If you have a feature that gives you a bonus action, you can take it on your turn. You choose when to take a bonus action during your turn, unless the bonus action's timing is specified."

## Confidence
**High** — The SRD explicitly states the player chooses when to take a bonus action during their turn."""

def ask_llm(query, retrieved):
    # Reorder chunks for better LLM attention
    retrieved = reorder_for_attention(retrieved)
    parts = []
    for c, s in retrieved:
        header = c.get("header", "")
        pages = c.get("pages", [0])
        page_str = ", ".join(str(p) for p in pages)
        parts.append(f"[Section: {header}, Pages: {page_str}]\n{c['text']}")
    context = "\n\n---\n\n".join(parts)
    prompt = SYSTEM_PROMPT + "\n\nRETRIEVED SRD SECTIONS:\n" + context + "\n\nQUESTION: " + query
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0)
    )
    return response.text

# --- Streamlit UI ---
st.title("D&D 5e Rules Assistant")
st.caption("Ask any rule question — answers cite the official SRD 5.2 with page numbers (hybrid semantic + keyword search)")

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
