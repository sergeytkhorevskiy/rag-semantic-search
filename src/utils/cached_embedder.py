import os
import numpy as np
from typing import List
from src.utils.embedding_cache import EmbeddingCache
from src.ingest.embed import E5Embedder

def _normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype="float32")
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n

class CachedEmbedder:
    """Wraps E5Embedder with a disk cache.
    Set EMB_CACHE=false to disable, EMB_CACHE_PATH to change location.
    """
    def __init__(self, model_name: str, cache_path: str | None = None):
        self.model_name = model_name
        self.inner = E5Embedder(model_name)
        if cache_path is None:
            cache_path = os.getenv("EMB_CACHE_PATH", "./index/emb_cache.sqlite3")
        self.cache = EmbeddingCache(cache_path)

    def _embed_with_cache(self, texts: List[str], fn):
        texts = [t if isinstance(t, str) else str(t) for t in texts]
        cached = self.cache.get_many(self.model_name, texts)
        to_compute_idx = [i for i, v in enumerate(cached) if v is None]
        computed_vectors = []
        if to_compute_idx:
            batch = [texts[i] for i in to_compute_idx]
            arr = fn(batch)
            arr = _normalize(arr)
            computed_vectors = arr.tolist()
            self.cache.put_many(self.model_name, batch, computed_vectors)
        # stitch together
        out = []
        ci = 0
        for i in range(len(texts)):
            if cached[i] is not None:
                out.append([float(x) for x in cached[i]])
            else:
                out.append(computed_vectors[ci])
                ci += 1
        return np.array(out, dtype="float32")

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        return self._embed_with_cache(queries, self.inner.embed_queries)

    def embed_passages(self, passages: List[str]) -> np.ndarray:
        return self._embed_with_cache(passages, self.inner.embed_passages)

def get_embedder(model_name: str):
    use_cache = os.getenv("EMB_CACHE", "true").lower() != "false"
    if use_cache:
        return CachedEmbedder(model_name)
    return E5Embedder(model_name)
