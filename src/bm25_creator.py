
# bm25_creating.py reads all chunks from SQLite then tokenizes them then builds a BM25 keyword index and then saves it to bm25.pkl.
# Later, queries can be scored against this BM25 index for keyword-based relevance.


import sqlite3, pickle
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

DB_PATH = "./db/chunks.db"

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("SELECT id, d_chunk FROM chunks ORDER BY id")
rows = c.fetchall()
conn.close()

ids = [r[0] for r in rows]
docs = [r[1] for r in rows]
tokenized = [word_tokenize(doc.lower()) for doc in docs]

bm25 = BM25Okapi(tokenized)

with open("./db/bm25.pkl", "wb") as f:
    pickle.dump({"bm25": bm25, "ids": ids, "tokenized": tokenized}, f)

print("BM25 built for", len(docs), "docs")
