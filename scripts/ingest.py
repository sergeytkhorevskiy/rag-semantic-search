import os, json, sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from src.ingest.extract import load_documents
from src.ingest.chunk import make_chunks
from src.ingest.build_index import build_faiss
from src.utils.cached_embedder import get_embedder

load_dotenv()

RAW = os.getenv("RAW_DATA_DIR", "data/raw")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "./index/chunks.jsonl")
INDEX_PATH  = os.getenv("INDEX_PATH", "./index/faiss.index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
EMB_CACHE   = os.getenv("EMB_CACHE", "true").lower() != "false"

if __name__ == "__main__":
    print("[1/4] Loading documents...")
    docs = load_documents(RAW)
    print(f"Loaded {len(docs)} documents from {RAW}")

    print("[2/4] Chunking...")
    rows = make_chunks(docs, max_chars=1000, overlap=100)
    os.makedirs(os.path.dirname(CHUNKS_PATH), exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} chunks -> {CHUNKS_PATH}")

    print("[3/4] Pre-warming embedding cache..." if EMB_CACHE else "[3/4] Skipping cache warm-up (EMB_CACHE=false)")
    if EMB_CACHE and rows:
        embedder = get_embedder(EMBED_MODEL)
        texts = [r["text"] for r in rows]
        # Batch to avoid OOM
        B = int(os.getenv("EMB_WARM_BATCH", "256"))
        for i in range(0, len(texts), B):
            _ = embedder.embed_passages(texts[i:i+B])
        print("Embedding cache warmed.")

    print("[4/4] Building FAISS index...")
    count, _ = build_faiss(CHUNKS_PATH, INDEX_PATH, EMBED_MODEL)
    print(f"Indexed {count} chunks -> {INDEX_PATH}")
