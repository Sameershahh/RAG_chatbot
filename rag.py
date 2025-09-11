import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline


INDEX_DIR = "index"
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# Load FAISS index
index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
with open(os.path.join(INDEX_DIR, "metadata.json"), "r", encoding="utf-8") as f:
metadata = json.load(f)


# Answer generator fallback (local model)
generator = pipeline("text-generation", model="gpt2")


def retrieve(query, top_k=3):
query_vec = embedder.encode([query], convert_to_numpy=True)
scores, idxs = index.search(query_vec, top_k)
results = [(float(scores[0][i]), metadata[idxs[0][i]]) for i in range(top_k)]
return results


def generate_answer_with_context(retrieved, query):
context = "\n".join([r[1]["chunk_text"] for r in retrieved])
prompt = f"Answer the question using the following context:\n{context}\n\nQuestion: {query}\nAnswer:"


# Use local HF model for now
output = generator(prompt, max_length=250, do_sample=True, temperature=0.7)[0]["generated_text"]
return output, "huggingface-gpt2"