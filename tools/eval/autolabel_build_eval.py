import os, sys, json, re, math
from pathlib import Path
from collections import Counter
from typing import List

_WORD = re.compile(r"\w+", re.UNICODE)
def toks(s: str) -> list[str]:
    return [w for w in _WORD.findall((s or '').lower()) if len(w) >= 2]

class BM25:
    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = corpus_tokens
        self.N = len(corpus_tokens)
        self.doc_len = [len(x) for x in corpus_tokens]
        self.avgdl = (sum(self.doc_len)/self.N) if self.N else 0.0
        self.tf = [Counter(x) for x in corpus_tokens]
        df = Counter()
        for doc in corpus_tokens:
            for t in set(doc):
                df[t] += 1
        self.idf = {t: math.log(1 + (self.N - df_t + 0.5)/(df_t + 0.5)) for t, df_t in df.items()}
    def score(self, q: list[str], idx: int) -> float:
        dl = self.doc_len[idx] or 1
        tf = self.tf[idx]
        sc = 0.0
        for t in q:
            if t not in tf: continue
            idf = self.idf.get(t, 0.0)
            f = tf[t]
            denom = f + self.k1 * (1 - self.b + self.b * dl/(self.avgdl or 1.0))
            sc += idf * (f * (self.k1 + 1) / (denom or 1.0))
        return sc
    def topk(self, q: list[str], k: int) -> list[int]:
        scores = [self.score(q, i) for i in range(self.N)]
        idxs = sorted(range(self.N), key=lambda i: scores[i], reverse=True)[:k]
        return idxs, [scores[i] for i in idxs]

def load_chunks(chunks_path: str):
    rows = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    texts = [r.get('text','') for r in rows]
    tokens = [toks(t) for t in texts]
    return rows, tokens

def build(eval_seed_path: str, chunks_path: str, out_path: str, top_k: int = 6):
    rows, tokens = load_chunks(chunks_path)
    bm25 = BM25(tokens)
    with open(eval_seed_path, 'r', encoding='utf-8') as f:
        seeds = [json.loads(l) for l in f if l.strip()]
    out_lines = []
    for i, s in enumerate(seeds, 1):
        qid = str(s.get('id', f'q{i}'))
        q = s['question']
        answers = s.get('answers', [])
        q_terms = s.get('keywords')
        q_tokens = toks(' '.join(q_terms)) if q_terms else toks(q)
        idxs, _ = bm25.topk(q_tokens, top_k)
        rel_chunk_ids = [rows[j]['chunk_id'] for j in idxs]
        out_lines.append({
            "id": qid, "question": q, "answers": answers,
            "relevant_chunk_ids": rel_chunk_ids
        })
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for o in out_lines:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out_lines)} items -> {out_path}")
if __name__ == "__main__":
    import argparse, os
    p = argparse.ArgumentParser()
    p.add_argument("--seed_path", default="data/eval/questions_seed.jsonl")
    p.add_argument("--chunks_path", default=os.getenv("CHUNKS_PATH", "./index/chunks.jsonl"))
    p.add_argument("--out_path", default="data/eval/qa.jsonl")
    p.add_argument("--top_k", type=int, default=6)
    args = p.parse_args()
    build(args.seed_path, args.chunks_path, args.out_path, top_k=args.top_k)
