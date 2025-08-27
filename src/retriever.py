import json
from typing import List, Dict, Tuple
import faiss
import numpy as np
from src.utils.cached_embedder import get_embedder
from src.search.bm25 import BM25Okapi, tokenize

def _normalize_scores(m: dict) -> dict:
    if not m:
        return {}
    vals = list(m.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return {k: 1.0 for k in m}
    return {k: (v - lo)/(hi - lo) for k, v in m.items()}

class Retriever:
    def __init__(self, index_path: str, chunks_path: str, embed_model: str):
        # Vector index
        self.index = faiss.read_index(index_path)
        # Chunks
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.rows = [json.loads(l) for l in f.read().splitlines()]
        # Embedder (cached)
        self.embedder = get_embedder(embed_model)
        # BM25 over chunk texts
        self.texts = [r.get("text", "") for r in self.rows]
        self.tokens = [tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(self.tokens)

    # ---------- Vector only ----------
    def _vector_topk(self, query: str, top_k: int) -> Tuple[List[Dict], List[int], List[float]]:
        q = self.embedder.embed_queries([query]).astype("float32")
        D, I = self.index.search(q, top_k)
        out = []
        idxs = I[0].tolist()
        scores = D[0].tolist()
        for score, idx in zip(scores, idxs):
            row = self.rows[idx]
            out.append({
                "score": float(score),
                "text": row["text"],
                "chunk_id": row["chunk_id"],
                "doc_path": row["doc_path"],
                "mode": "vector"
            })
        return out, idxs, scores

    # ---------- BM25 only ----------
    def _bm25_topk(self, query: str, top_k: int) -> Tuple[List[Dict], List[int], List[float]]:
        q_tokens = tokenize(query)
        scores_list = self.bm25.get_scores(q_tokens)
        idxs = np.argsort(scores_list)[::-1][:top_k].tolist()
        out = []
        scores = []
        for idx in idxs:
            row = self.rows[int(idx)]
            sc = float(scores_list[int(idx)])
            scores.append(sc)
            out.append({
                "score": sc,
                "text": row["text"],
                "chunk_id": row["chunk_id"],
                "doc_path": row["doc_path"],
                "mode": "bm25"
            })
        return out, idxs, scores

    # ---------- MMR (vector-only diversity) ----------
    def _mmr(self, query: str, cand_idxs: List[int], top_k: int = 8, lambda_mult: float = 0.6) -> List[int]:
        q_emb = self.embedder.embed_queries([query]).astype("float32")[0]
        cand_texts = [self.rows[i]["text"] for i in cand_idxs]
        cand_embs = self.embedder.embed_passages(cand_texts)
        selected = []
        selected_idx = []
        def sim(a, b):  # cosine since normalized
            return float(np.dot(a, b))
        while len(selected) < min(top_k, len(cand_idxs)):
            best_j = None
            best_score = -1e9
            for j in range(len(cand_idxs)):
                if j in selected_idx:
                    continue
                relevance = sim(q_emb, cand_embs[j])
                diversity = 0.0
                if selected_idx:
                    diversity = max(sim(cand_embs[j], cand_embs[k]) for k in selected_idx)
                mmr = lambda_mult * relevance - (1 - lambda_mult) * diversity
                if mmr > best_score:
                    best_score = mmr
                    best_j = j
            selected_idx.append(best_j)
            selected.append(cand_idxs[best_j])
        return selected

    # ---------- Lexical overlap heuristic ----------
    def _lexical_overlap_ratio(self, q_tokens: List[str], idxs: List[int], check_k: int = 8) -> float:
        idxs = idxs[:max(1, min(len(idxs), check_k))]
        if not idxs or not q_tokens:
            return 0.0
        overlaps = 0
        total = 0
        qset = set(q_tokens)
        for i in idxs:
            tset = set(self.tokens[i])
            inter = qset.intersection(tset)
            overlaps += len(inter)
            total += max(1, len(qset))
        return overlaps / float(total)

    # ---------- Hybrid with optional lexical fallback ----------
    def _hybrid(self, query: str, top_k: int = 8, fetch_k: int = 64, alpha: float = 0.6, mmr: bool = False, lambda_mult: float = 0.6, lexical_fallback: bool = True, fallback_check_k: int = 12) -> List[Dict]:
        # 1) Vector candidates
        q = self.embedder.embed_queries([query]).astype("float32")
        Dv, Iv = self.index.search(q, fetch_k)
        vec_scores = {int(i): float(s) for i, s in zip(Iv[0].tolist(), Dv[0].tolist())}

        # 2) BM25 candidates
        q_tokens = tokenize(query)
        bm_scores_list = self.bm25.get_scores(q_tokens)
        Ibm = np.argsort(bm_scores_list)[::-1][:fetch_k].tolist()
        bm_scores = {int(i): float(bm_scores_list[i]) for i in Ibm}

        # 3) Adaptive alpha via lexical overlap
        alpha_used = alpha
        if lexical_fallback:
            overlap_ratio = self._lexical_overlap_ratio(q_tokens, Iv[0].tolist(), check_k=fallback_check_k)
            # low overlap => rely more on BM25
            if overlap_ratio < 0.15:
                alpha_used = min(alpha_used, 0.3)

        # 4) Union & normalize both score spaces
        cand_idxs = list(set(list(vec_scores.keys()) + list(bm_scores.keys())))
        if not cand_idxs:
            return []
        vec_n = _normalize_scores(vec_scores)
        bm_n = _normalize_scores(bm_scores)

        # 5) Combine
        combined = {}
        for i in cand_idxs:
            v = vec_n.get(i, 0.0)
            b = bm_n.get(i, 0.0)
            combined[i] = alpha_used * v + (1.0 - alpha_used) * b

        # 6) Take top and (optionally) run MMR by embeddings
        top_idxs = sorted(combined.keys(), key=lambda k: combined[k], reverse=True)[:max(top_k, 2)]
        if mmr:
            top_idxs = self._mmr(query, top_idxs, top_k=top_k, lambda_mult=lambda_mult)
        else:
            top_idxs = top_idxs[:top_k]

        out = []
        for idx in top_idxs:
            row = self.rows[int(idx)]
            out.append({
                "score": float(combined[idx]),
                "text": row["text"],
                "chunk_id": row["chunk_id"],
                "doc_path": row["doc_path"],
                "mode": "hybrid-fallback" if lexical_fallback and alpha_used != alpha else "hybrid",
            })
        return out

    # Public API
    def search(self, query: str, top_k: int = 8, mode: str = "vector", mmr: bool = False, fetch_k: int = 64, alpha: float = 0.6, lexical_fallback: bool = True) -> List[Dict]:
        mode = (mode or "vector").lower()
        if mode == "bm25":
            hits, _, _ = self._bm25_topk(query, top_k)
            return hits
        if mode == "hybrid":
            return self._hybrid(query, top_k=top_k, fetch_k=fetch_k, alpha=alpha, mmr=mmr, lexical_fallback=lexical_fallback)
        # default: vector
        hits, _, _ = self._vector_topk(query, top_k)
        return hits
