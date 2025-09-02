import os
import json
import sys
import asyncio
import aiofiles
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from concurrent.futures import ThreadPoolExecutor

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from src.ingest.extract import EXT_READERS
from src.ingest.chunk import chunk_text
from src.ingest.build_index import build_faiss
from src.utils.cached_embedder import get_embedder

load_dotenv()

RAW = os.getenv("RAW_DATA_DIR", "data/raw")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "./index/chunks.jsonl")
INDEX_PATH = os.getenv("INDEX_PATH", "./index/faiss.index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
EMB_CACHE = os.getenv("EMB_CACHE", "true").lower() != "false"
N_WORKERS = int(os.getenv("N_WORKERS", str(max(1, cpu_count() - 1))))

# Import document readers
from src.ingest.extract import read_pdf, read_html, read_md_or_txt

async def read_file_async(path, reader_func):
    """Read file asynchronously using thread pool"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(executor, reader_func, path)

async def process_document_async(file_info):
    """Process a single document asynchronously"""
    dirpath, filename = file_info
    ext = os.path.splitext(filename.lower())[1]
    if ext not in EXT_READERS:
        return None
    
    path = os.path.join(dirpath, filename)
    try:
        if ext == ".pdf":
            text = await read_file_async(path, read_pdf)
        elif ext in [".html", ".htm"]:
            text = await read_file_async(path, read_html)
        else:  # .md, .txt
            text = await read_file_async(path, read_md_or_txt)
        
        if text and text.strip():
            return {"path": path, "text": text}
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
    return None

def get_document_files(root):
    """Get list of document files to process"""
    files_to_process = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            ext = os.path.splitext(filename.lower())[1]
            if ext in EXT_READERS:
                files_to_process.append((dirpath, filename))
    return files_to_process

async def process_documents_async(root, max_concurrent=10):
    """Load documents using async processing"""
    files_to_process = get_document_files(root)
    print(f"Found {len(files_to_process)} documents to process")
    
    if not files_to_process:
        return []
    
    # Process documents with limited concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(file_info):
        async with semaphore:
            return await process_document_async(file_info)
    
    tasks = [process_with_semaphore(file_info) for file_info in files_to_process]
    results = await asyncio.gather(*tasks)
    
    # Filter out None results
    docs = [doc for doc in results if doc is not None]
    return docs

def chunk_document_parallel(doc, max_chars=1000, overlap=100):
    """Chunk a single document in parallel"""
    parts = chunk_text(doc["text"], max_chars=max_chars, overlap=overlap)
    chunks = []
    for j, part in enumerate(parts):
        chunks.append({
            "doc_path": doc["path"],
            "chunk_id": f"{doc['path']}::chunk_{j}",
            "text": part,
        })
    return chunks

def process_chunks_parallel(docs, max_chars=1000, overlap=100, n_workers=None):
    """Process chunks using parallel processing"""
    if n_workers is None:
        n_workers = N_WORKERS
    
    print(f"Chunking {len(docs)} documents with {n_workers} workers")
    
    # Create partial function with fixed parameters
    chunk_func = partial(chunk_document_parallel, max_chars=max_chars, overlap=overlap)
    
    # Use multiprocessing to chunk documents in parallel
    with Pool(processes=n_workers) as pool:
        chunk_results = pool.map(chunk_func, docs)
    
    # Flatten results
    all_chunks = []
    for chunks in chunk_results:
        all_chunks.extend(chunks)
    
    return all_chunks

async def save_chunks_async(chunks, filepath):
    """Save chunks asynchronously"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    async with aiofiles.open(filepath, "w", encoding="utf-8") as f:
        for chunk in chunks:
            await f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

def warm_embedding_cache_parallel(texts, embedder, batch_size=256):
    """Warm embedding cache using parallel processing"""
    print(f"Warming embedding cache for {len(texts)} texts")
    
    # Process in batches to avoid OOM
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        _ = embedder.embed_passages(batch)
        print(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")

async def ingest_async():
    """Main ingestion function with async processing"""
    start_time = time.time()
    
    print(f"[1/4] Loading documents asynchronously...")
    docs = await process_documents_async(RAW, max_concurrent=10)
    print(f"Loaded {len(docs)} documents from {RAW}")
    
    print(f"[2/4] Chunking with {N_WORKERS} workers...")
    rows = process_chunks_parallel(docs, max_chars=1000, overlap=100)
    
    # Save chunks asynchronously
    print(f"Saving {len(rows)} chunks...")
    await save_chunks_async(rows, CHUNKS_PATH)
    print(f"Saved chunks -> {CHUNKS_PATH}")
    
    print("[3/4] Pre-warming embedding cache..." if EMB_CACHE else "[3/4] Skipping cache warm-up (EMB_CACHE=false)")
    if EMB_CACHE and rows:
        embedder = get_embedder(EMBED_MODEL)
        texts = [row["text"] for row in rows]
        batch_size = int(os.getenv("EMB_WARM_BATCH", "256"))
        warm_embedding_cache_parallel(texts, embedder, batch_size=batch_size)
        print("Embedding cache warmed.")
    
    print("[4/4] Building FAISS index...")
    count, _ = build_faiss(CHUNKS_PATH, INDEX_PATH, EMBED_MODEL)
    print(f"Indexed {count} chunks -> {INDEX_PATH}")
    
    total_time = time.time() - start_time
    print(f"✅ Async ingestion completed in {total_time:.2f} seconds")

def ingest():
    """Wrapper function to run async ingestion"""
    asyncio.run(ingest_async())

if __name__ == "__main__":
    print(f"🚀 Starting async ingestion with {N_WORKERS} workers")
    ingest()

