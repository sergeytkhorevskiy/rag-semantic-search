from typing import List, Dict
import re

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZА-ЯІЇЄҐ0-9])", re.UNICODE)

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    text = " ".join(text.split())
    # sentence-aware split, then greedy pack
    sents = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + 1 + len(s) <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    # add overlap via sliding window
    if overlap > 0 and chunks:
        windowed = []
        for i, ch in enumerate(chunks):
            if i == 0:
                windowed.append(ch)
            else:
                prev_tail = chunks[i-1][-overlap:]
                windowed.append((prev_tail + " " + ch).strip())
        chunks = windowed
    return chunks

def make_chunks(docs: List[Dict], **kw) -> List[Dict]:
    rows = []
    for d in docs:
        parts = chunk_text(d["text"], **kw)
        for j, p in enumerate(parts):
            rows.append({
                "doc_path": d["path"],
                "chunk_id": f"{d['path']}::chunk_{j}",
                "text": p,
            })
    return rows
