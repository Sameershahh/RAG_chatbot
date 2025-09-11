import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ----------------------------
# Settings
# ----------------------------
DOCS_DIR = "docs"  # folder containing .txt files
INDEX_PATH = "faiss.index"
META_PATH = "metadata.json"
CHUNK_SIZE = 500  # characters per chunk

# ----------------------------
# Load embedding model
# ----------------------------
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------
# Helper function to split text
# ----------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# ----------------------------
# Process documents
# ----------------------------
chunks = []
for fname in os.listdir(DOCS_DIR):
    if fname.endswith(".txt"):
        path = os.path.join(DOCS_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            for chunk in chunk_text(text):
                chunks.append({"text": chunk, "source": fname})

print(f"Loaded {len(chunks)} chunks from {len(os.listdir(DOCS_DIR))} documents.")

# ----------------------------
# Compute embeddings
# ----------------------------
texts = [c["text"] for c in chunks]
embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
dim = embeddings.shape[1]

# ----------------------------
# Build FAISS index
# ----------------------------
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings, dtype=np.float32))
faiss.write_index(index, INDEX_PATH)

# ----------------------------
# Save metadata
# ----------------------------
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"✅ Index and metadata saved!")
