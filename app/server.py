import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, Query
from pydantic import BaseModel
from dotenv import load_dotenv

from src.retriever import Retriever
from src.rag import answer_with_citations
try:
    from src.rerank import Reranker
except Exception:
    Reranker = None

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
INDEX_PATH = os.getenv("INDEX_PATH", "./index/faiss.index")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "./index/chunks.jsonl")
RE_RANK = os.getenv("RE_RANK", "false").lower() == "true"
USE_MMR = os.getenv("USE_MMR", "false").lower() == "true"

# Hybrid defaults
SEARCH_MODE = os.getenv("SEARCH_MODE", "hybrid")
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.65"))
FETCH_K = int(os.getenv("FETCH_K", "64"))
LEXICAL_FALLBACK = os.getenv("LEXICAL_FALLBACK", "true").lower() != "false"

app = FastAPI(title="RAG Semantic Search API (Hybrid+Cache+Fallback)")

retriever = Retriever(INDEX_PATH, CHUNKS_PATH, EMBED_MODEL)
reranker = Reranker() if (RE_RANK and Reranker is not None) else None

class AskRequest(BaseModel):
    question: str
    top_k: int = 8
    mmr: Optional[bool] = None
    mode: Optional[str] = None
    alpha: Optional[float] = None
    fetch_k: Optional[int] = None
    lexical_fallback: Optional[bool] = None

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/search")
async def search(
    q: str = Query(..., description="query"),
    top_k: int = 8,
    mmr: bool = False,
    mode: str = SEARCH_MODE,
    alpha: float = HYBRID_ALPHA,
    fetch_k: int = FETCH_K,
    lexical_fallback: bool = LEXICAL_FALLBACK,
):
    try:
        hits = retriever.search(q, top_k=top_k, mode=mode, mmr=mmr or USE_MMR, fetch_k=fetch_k, alpha=alpha, lexical_fallback=lexical_fallback)
        if reranker and mode != "bm25":  # rerank makes sense for vector/hybrid
            try:
                hits = reranker.rerank(q, hits, top_k)
            except Exception as e:
                print(f"Reranking failed: {e}")
        return {"mode": mode, "alpha": alpha, "lexical_fallback": lexical_fallback, "hits": hits}
    except Exception as e:
        return {"error": str(e), "hits": []}

@app.post("/ask")
async def ask(req: AskRequest):
    try:
        use_mmr = req.mmr if req.mmr is not None else USE_MMR
        mode = (req.mode or SEARCH_MODE)
        alpha = float(req.alpha if req.alpha is not None else HYBRID_ALPHA)
        fetch_k = int(req.fetch_k if req.fetch_k is not None else FETCH_K)
        lexical_fb = req.lexical_fallback if req.lexical_fallback is not None else LEXICAL_FALLBACK

        hits = retriever.search(req.question, top_k=req.top_k, mode=mode, mmr=use_mmr, fetch_k=fetch_k, alpha=alpha, lexical_fallback=lexical_fb)
        if reranker and mode != "bm25":  # rerank makes sense for vector/hybrid
            try:
                hits = reranker.rerank(req.question, hits, req.top_k)
            except Exception as e:
                print(f"Reranking failed: {e}")
        ans = answer_with_citations(req.question, hits)
        return {"answer": ans["answer"], "mode": mode, "alpha": alpha, "lexical_fallback": lexical_fb, "hits": hits}
    except Exception as e:
        return {"error": str(e), "answer": "An error occurred while processing your request.", "hits": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
