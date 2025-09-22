
# embeddings.py work is to generate embeddings and build a FAISS index
# Later, so when you ask a question to system it can use this FAISS index to quickly find the most similar chunks.


import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DB_PATH = "./db/chunks.db"
EMB_FILE = "./db/embeddings.npy"
IDS_FILE = "./db/ids.npy"
INDEX_FILE = "./db/faiss.index"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Load chunks ---
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("SELECT id, d_chunk FROM chunks ORDER BY id")
rows = c.fetchall()
conn.close()

ids = [r[0] for r in rows]
texts = [r[1] for r in rows]
print(f"Loaded {len(texts)} chunks from DB")

# --- Build embeddings ---
model = SentenceTransformer(MODEL_NAME)
embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

# Normalize for cosine similarity
norms = np.linalg.norm(embs, axis=1, keepdims=True)
embs = embs / np.where(norms == 0, 1, norms)

# --- Save numpy arrays ---
np.save(EMB_FILE, embs)
np.save(IDS_FILE, np.array(ids))
print(f"Saved {EMB_FILE}, {IDS_FILE}")

# --- Build FAISS index ---
d = embs.shape[1]
index = faiss.IndexFlatIP(d)  
index.add(embs.astype("float32"))
faiss.write_index(index, INDEX_FILE)
print(f"Saved {INDEX_FILE}, total vectors: {index.ntotal}")
