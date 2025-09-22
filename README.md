# Q&A Service over Industrial Safety PDFs  

## Data Preparation  

The dataset consists of **20 PDFs** on industrial & machine safety, with a `sources.json` mapping each file to its title and URL.  

Some issues were present in the provided data:  
- A few PDFs in the ZIP did not exactly match the titles/URLs listed in `sources.json`.  
- For some mismatched cases, I re-downloaded the correct PDF directly from the URL in `sources.json`.  
- During ingestion, I used a systematic loop to pair each PDF with its corresponding title and URL before splitting into chunks. 

To make things easier, I also stored all processed data (chunks, embeddings, FAISS index, BM25 index, etc.) inside the db/ folder.
This way, you can directly run the program without having to rebuild everything from scratch, though the scripts are included if you’d like to regenerate. 

This ensured consistency between documents and metadata.  

---

## Setup  

1. **Clone the repo & install dependencies**  
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   pip install -r requirements.txt
   ```

2. **Prepare the database**  
   ```bash
   # Step 1: Ingest PDFs into chunks
   python ingestion.py  

   # Step 2: Build embeddings + FAISS index
   python build_embeddings.py  

   # Step 3: Build BM25 index
   python build_bm25.py  
   ```

3. **Run the API**  
   ```bash
   uvicorn main:app --reload
   ```
   The API will start at: [http://127.0.0.1:8000](http://127.0.0.1:8000)  


---

## What I Learned  

Through this project, I built a small but complete **retrieval-augmented QA system**.  
- Splitting documents into paragraph-sized chunks and storing them in SQLite made retrieval simple and efficient.  
- Using **FAISS + sentence-transformer embeddings** provided a strong baseline for semantic search.  
- Adding a **BM25 hybrid reranker** improved tricky queries where pure semantic similarity was not enough.  
- Implementing an **abstain mechanism** avoided hallucinated answers when the system did not have enough confidence.  

I also learned how to handle imperfect datasets — reconciling mismatched filenames with `sources.json` and downloading missing files ensured the pipeline was robust and reproducible.  

---

## Example Requests  

**Easy (baseline mode):**  
```bash
curl -X POST "http://127.0.0.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"q":"What is machine guarding?","k":2,"mode":"baseline"}'
```

**Tricky (hybrid mode):**  
```bash
curl -X POST "http://127.0.0.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"q":"What is the difference between SIL and PL in machine safety?","k":2,"mode":"hybrid"}'
```

**I’ve done my best to make this project complete and suitable for review. If there are any issues, please let me know — I’d be happy to fix them. I really enjoyed working on this assignment and I’m excited about the opportunity to work with your team.**
