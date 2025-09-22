
# baseline_search.py implements the baseline QA retrieval step using embeddings + FAISS.
# It returns the most semantically similar chunks to a query, ranked by cosine similarity.


import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DB_PATH = "./db/chunks.db"
EMB_FILE = "./db/embeddings.npy"
IDS_FILE = "./db/ids.npy"
INDEX_FILE = "./db/faiss.index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Load resources ---
embs = np.load(EMB_FILE)
ids = np.load(IDS_FILE)
index = faiss.read_index(INDEX_FILE)
model = SentenceTransformer(MODEL_NAME)

def baseline_search(query, top_k=3):
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    D, I = index.search(q_emb.astype("float32"), top_k)

    results = []
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for rank, idx in enumerate(I[0]):
        chunk_id = int(ids[idx])
        c.execute("SELECT d_title, d_url, d_chunk FROM chunks WHERE id=?", (chunk_id,))
        title, url, chunk = c.fetchone()
        results.append({
            "rank": rank + 1,
            "score": float(D[0][rank]),
            "title": title,
            "url": url,
            "chunk": chunk[:250] + "..."
        })
    conn.close()
    return results

# --- Demo ---
if __name__ == "__main__":
    query = "What is machine guarding?"
    results = baseline_search(query, top_k=3)
    for r in results:
        print(f"\nRank {r['rank']} | Score: {r['score']:.4f}")
        print(f"Title: {r['title']}")
        print(f"URL  : {r['url']}")
        print(f"Chunk: {r['chunk']}")
