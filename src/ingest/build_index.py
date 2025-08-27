import json
import os
import faiss
import numpy as np
from .embed import E5Embedder

def build_faiss(chunks_path: str, index_path: str, embed_model: str):
    with open(chunks_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    texts = [r["text"] for r in rows]
    embedder = E5Embedder(embed_model)
    X = embedder.embed_passages(texts)

    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
    index.add(X)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

    return len(rows), rows
