import streamlit as st
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ----------------------------
# Settings
# ----------------------------
INDEX_PATH = "faiss.index"
META_PATH = "metadata.json"

# ----------------------------
# Load FAISS index and metadata
# ----------------------------
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# ----------------------------
# Models
# ----------------------------
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
generator = pipeline("text-generation", model="gpt2", device=-1)  # device=-1 for CPU

# ----------------------------
# Helper functions
# ----------------------------
def retrieve(query, top_k=3):
    """Retrieve top-k chunks for a query."""
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb, dtype=np.float32), top_k)
    return [(metadata[i], D[0][j]) for j, i in enumerate(I[0])]

def rag_answer(query):
    """Generate answer using retrieved context + LLM."""
    retrieved = retrieve(query, top_k=3)
    # extract only text
    context_texts = [chunk["text"] if isinstance(chunk, dict) else chunk for chunk, _ in retrieved]
    context = "\n".join(context_texts)

    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = generator(prompt, max_length=200, num_return_sequences=1)[0]["generated_text"]

    return response, retrieved

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🧠 RAG Chatbot")
st.write("Ask me anything based on your knowledge base!")

query = st.text_input("Enter your question:")

if query:
    answer, retrieved = rag_answer(query)

    st.subheader("💡 Answer")
    st.write(answer)

    st.subheader("📚 Retrieved Context & Sources")
    for i, (chunk, score) in enumerate(retrieved, 1):
        text = chunk["text"] if isinstance(chunk, dict) else chunk
        source = chunk.get("source", "unknown") if isinstance(chunk, dict) else "unknown"
        st.markdown(f"**{i}.** {text} _(Source: {source}, Score: {score:.3f})_")
