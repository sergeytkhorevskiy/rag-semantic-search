from typing import List, Dict
import numpy as np
import torch
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CrossEncoder = None
    CROSS_ENCODER_AVAILABLE = False

class SimpleReranker:
    """Simple fallback reranker using term matches and base score."""
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        q_terms = query.lower().split()
        scored = []
        for c in candidates:
            t = c.get("text","").lower()
            term_matches = sum(1 for term in q_terms if term in t)
            c2 = dict(c)
            c2["rerank_score"] = float(c2.get("score", 0.0)) * (1.0 + 0.1*term_matches)
            scored.append(c2)
        return sorted(scored, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

class Reranker:
    def __init__(self, model_name: str = "jinaai/jina-reranker-v2-base-multilingual"):
        if not CROSS_ENCODER_AVAILABLE:
            raise ImportError("Install sentence-transformers for CrossEncoder reranker")
        try:
            self.model = CrossEncoder(model_name, trust_remote_code=True, device="cpu")
            if hasattr(self.model, "model"):
                self.model.model = self.model.model.float()
            self.use_cross_encoder = True
        except Exception as e:
            print(f"CrossEncoder init failed: {e}. Falling back to SimpleReranker.")
            self.use_cross_encoder = False
            self.simple = SimpleReranker()

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        if not self.use_cross_encoder:
            return self.simple.rerank(query, candidates, top_k)
        try:
            pairs = [(query, c.get("text","")) for c in candidates]
            with torch.no_grad():
                scores = self.model.predict(pairs)
            if isinstance(scores, torch.Tensor):
                scores = scores.float().cpu().numpy()
            elif isinstance(scores, np.ndarray):
                scores = scores.astype(np.float32)
            else:
                scores = np.array(scores, dtype=np.float32)
            out = []
            for c, s in zip(candidates, list(scores)):
                c2 = dict(c)
                c2["rerank_score"] = float(s)
                out.append(c2)
            return sorted(out, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
        except Exception as e:
            print(f"CrossEncoder reranking failed: {e}. Falling back to SimpleReranker.")
            return self.simple.rerank(query, candidates, top_k)
