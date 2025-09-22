# answer_to_json.py is used to generate you results of 8 questions in results.json

import json
from src.baseline_search import baseline_search
from src.hybrid_search import hybrid_search

THRESHOLD = 0.3 

def get_score(ctx):
    for k in ("final_score", "score", "cos_score", "bm25_score"):
        if k in ctx and ctx[k] is not None:
            return float(ctx[k])
    return None

with open("questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

results = []

for q in questions:
    query = q["q"]
    b = baseline_search(query, top_k=1)
    if b:
        score = get_score(b[0])
        if score is not None and score >= THRESHOLD:
            b_title = b[0]["title"]
            b_answer = b[0]["chunk"]
        else:
            b_title = "ABSTAIN"
            b_answer = None
    else:
        b_title = "ABSTAIN"
        b_answer = None
        
    h = hybrid_search(query, top_k=1, candidate_k=30, alpha=0.6)
    if h:
        score = get_score(h[0])
        if score is not None and score >= THRESHOLD:
            h_title = h[0]["title"]
            h_answer = h[0]["chunk"]
        else:
            h_title = "ABSTAIN"
            h_answer = None
    else:
        h_title = "ABSTAIN"
        h_answer = None

    results.append({
        "question": query,
        "baseline_title": b_title,
        "baseline_answer": b_answer,
        "hybrid_title": h_title,
        "hybrid_answer": h_answer
    })

# Save results to JSON
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("✅ Results saved to results.json")
