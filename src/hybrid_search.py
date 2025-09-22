
# hybrid_search.py implements a hybrid reranker:
# Uses FAISS to get semantic candidates.
# Scores them again with BM25 (keyword relevance).
# Combines both scores with a weight (alpha).
# Returns the top reranked results with detailed scoring.


import faiss, numpy as np, sqlite3, pickle
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
import math

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_FILE = "./db/faiss.index"
EMB_FILE = "./db/embeddings.npy"
IDS_FILE = "./db/ids.npy"
BM25_FILE = "./db/bm25.pkl"
DB_PATH = "./db/chunks.db"

# in this i am load stuff 
index = faiss.read_index(INDEX_FILE)
embs = np.load(EMB_FILE)
ids = np.load(IDS_FILE) 
model = SentenceTransformer(MODEL)

with open(BM25_FILE, "rb") as f:
    bm25_data = pickle.load(f)
bm25 = bm25_data["bm25"]
bm25_ids = bm25_data["ids"]  

def _minmax_norm(arr):
    arr = np.array(arr, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.ones_like(arr) * 0.5
    return (arr - lo) / (hi - lo)

def hybrid_search(query, top_k=3, candidate_k=30, alpha=0.6):
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    D, I = index.search(q_emb.astype('float32'), candidate_k)
    cand_indices = I[0]   
    cos_scores = D[0].tolist()  
    q_tokens = word_tokenize(query.lower())
    all_bm25_scores = bm25.get_scores(q_tokens) 
    bm25_scores = [float(all_bm25_scores[idx]) for idx in cand_indices]
    cos_norm = _minmax_norm(cos_scores)
    bm25_norm = _minmax_norm(bm25_scores)
    final = alpha * cos_norm + (1 - alpha) * bm25_norm
    order = np.argsort(final)[::-1]
    results = []
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for rank_pos in order[:top_k]:
        emb_idx = cand_indices[rank_pos]
        chunk_db_id = int(ids[emb_idx])
        cur.execute("SELECT d_title, d_url, d_chunk FROM chunks WHERE id=?", (chunk_db_id,))
        row = cur.fetchone()
        results.append({
            "final_score": float(final[rank_pos]),
            "cos_score": float(cos_scores[rank_pos]),
            "bm25_score": float(bm25_scores[rank_pos]),
            "title": row[0],
            "url": row[1],
            "chunk": row[2]
        })

    conn.close()
    return results

# quick demo (you can test it)
if __name__ == "__main__":
    qs = "What is machine guarding?"
    out = hybrid_search(qs, top_k=3, candidate_k=50, alpha=0.6)
    for i,r in enumerate(out,1):
        print(i, r["final_score"], r["title"])
        print("  cos:", r["cos_score"], " bm25:", r["bm25_score"])
        print("  snippet:", r["chunk"][:200], "...\n")
