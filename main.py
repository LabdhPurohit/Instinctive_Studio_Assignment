from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from src.baseline_search import baseline_search  
from src.hybrid_search import hybrid_search      

app = FastAPI(title="Q&A Service")

THRESHOLD = 0.3 


class AskRequest(BaseModel):
    q: str
    k: int = 3
    mode: str = "baseline" 


class Context(BaseModel):
    score: Optional[float] = None   
    final_score: Optional[float] = None
    cos_score: Optional[float] = None
    bm25_score: Optional[float] = None
    title: str
    url: str
    chunk: Optional[str] = None  


class AskResponse(BaseModel):
    answer: Optional[str]
    contexts: List[Context]
    reranker_used: bool


def get_score(ctx: dict):
    for k in ("final_score", "score", "cos_score", "bm25_score"):
        if k in ctx and ctx[k] is not None:
            return float(ctx[k])
    return None


def generate_answer(query: str, contexts: List[dict]) -> Optional[str]:
    if not contexts:
        return None
    top = contexts[0]
    if top["title"] == "ABSTAIN":
        return None
    snippet = top["chunk"][:200]
    return f"{snippet} (Source: {top['title']}, {top['url']})"


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if req.mode == "baseline":
        results = baseline_search(req.q, top_k=req.k)
        rerank = False
    elif req.mode == "hybrid":
        results = hybrid_search(req.q, top_k=req.k, candidate_k=30, alpha=0.6)
        rerank = True
    else:
        return {"answer": None, "contexts": [], "reranker_used": False}

    # Apply abstain threshold
    filtered = []
    for r in results:
        score = get_score(r)
        if score is None or score < THRESHOLD:
            filtered.append({
                "score": score,
                "final_score": r.get("final_score"),
                "cos_score": r.get("cos_score"),
                "bm25_score": r.get("bm25_score"),
                "title": "ABSTAIN",
                "url": r.get("url", ""),
                "chunk": None
            })
        else:
            filtered.append(r)

    answer = generate_answer(req.q, filtered)
    return {
        "answer": answer,
        "contexts": filtered,
        "reranker_used": rerank
    }
