import os
import sqlite3
import hashlib
import zlib
import json
import threading
from typing import List, Optional

class EmbeddingCache:
    """Lightweight SQLite-based cache for embedding vectors.
    Stores (model, text) -> vector as compressed JSON in a BLOB.
    """
    def __init__(self, path: str = "./index/emb_cache.sqlite3"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # allow multithreaded use by retriever
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self.conn:
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    dims INTEGER NOT NULL,
                    vec  BLOB NOT NULL
                )"""
            )
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_model ON cache(model)")

    @staticmethod
    def _hash(model: str, text: str) -> str:
        return hashlib.sha256((model + "\n" + text).encode("utf-8")).hexdigest()

    def get_many(self, model: str, texts: List[str]):
        if not texts:
            return []
        keys = [self._hash(model, t) for t in texts]
        qmarks = ",".join(["?"] * len(keys))
        sql = f"SELECT key, vec, dims FROM cache WHERE key IN ({qmarks})"
        cur = self.conn.execute(sql, keys)
        rows = {k: (blob, dims) for (k, blob, dims) in cur.fetchall()}
        out = []
        for k in keys:
            if k in rows:
                blob, dims = rows[k]
                try:
                    vec = json.loads(zlib.decompress(blob).decode("utf-8"))
                except Exception:
                    vec = None
                out.append(vec)
            else:
                out.append(None)
        return out

    def put_many(self, model: str, texts: List[str], vectors: List[list]):
        if not texts:
            return
        rows = []
        for t, v in zip(texts, vectors):
            key = self._hash(model, t)
            try:
                blob = zlib.compress(json.dumps([float(x) for x in v]).encode("utf-8"))
            except Exception:
                # best-effort: skip if not serializable
                continue
            dims = len(v)
            rows.append((key, model, dims, blob))
        with self.lock:
            self.conn.executemany(
                "INSERT OR REPLACE INTO cache(key, model, dims, vec) VALUES (?,?,?,?)",
                rows
            )
            self.conn.commit()
