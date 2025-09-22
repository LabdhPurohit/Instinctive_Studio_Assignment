
# ingestion.py reads each PDF then splits it into chunks (350 words) and then stores the chunks in a SQLite database (chunks.db) 
# along with their title and URL and finally prints a preview.


import os
import sqlite3
import numpy as np
from PyPDF2 import PdfReader
import json

DB_PATH = "./db/chunks.db"

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    d_title TEXT,
    d_url TEXT,
    d_chunk TEXT
)
""")
conn.commit()


def creating_chunks(pdf, title, url, max_words=350):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"

    # Split by paragraphs
    paragraphs = text.split("\n\n")
    chunks, current = [], []

    for para in paragraphs:
        words = para.strip().split()
        if not words:
            continue
        if len(current) + len(words) > max_words:
            chunks.append(" ".join(current))
            current = []
        current.extend(words)

    if current:
        chunks.append(" ".join(current))

    return chunks

sources_file = './sources.json'
with open(sources_file, "r", encoding="utf-8") as f:
    sources = json.load(f)

for i, j in enumerate(sources):
    pdf_path = f"./pdfs/0{i+1}.pdf"
    chunks = creating_chunks(pdf_path, j['title'], j['url'])
    # Store in SQLite
    for chunk in chunks:
        c.execute("INSERT INTO chunks (d_title, d_url, d_chunk) VALUES (?, ?, ?)", 
                (j['title'], j['url'], chunk))
    conn.commit()
    print(f"Inserted {len(chunks)} chunks for {j['title']}")

# Count how many chunks stored
c.execute("SELECT COUNT(*) FROM chunks")
print("Total chunks:", c.fetchone()[0])

# Fetch a few chunks to preview
c.execute("SELECT id, d_title, d_url, substr(d_chunk, 1, 200) || '...' FROM chunks LIMIT 5")
rows = c.fetchall()

for row in rows:
    print("\nID:", row[0])
    print("Title:", row[1])
    print("URL:", row[2])
    print("Chunk:", row[3])

conn.close()