import numpy as np
from sentence_transformers import SentenceTransformer

class E5Embedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        self.model = SentenceTransformer(model_name)

    def embed_passages(self, texts):
        marked = [f"passage: {t}" for t in texts]
        embs = self.model.encode(marked, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
        return np.asarray(embs, dtype="float32")

    def embed_queries(self, texts):
        marked = [f"query: {t}" for t in texts]
        embs = self.model.encode(marked, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embs, dtype="float32")
