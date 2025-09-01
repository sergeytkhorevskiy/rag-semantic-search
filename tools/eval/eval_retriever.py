import os, json, sys
from pathlib import Path
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.retriever import Retriever
from tools.eval.metrics import precision_at_k, recall_at_k, mrr_at_k, ndcg_at_k_from_binary, average_precision

load_dotenv()
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
INDEX_PATH = os.getenv("INDEX_PATH", "./index/faiss.index")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "./index/chunks.jsonl")

@dataclass
class EvalItem:
    qid: str
    question: str
    relevant_chunks: List[str]
    relevant_docs: List[str]

def load_items(p: str) -> List[EvalItem]:
    items = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            items.append(EvalItem(
                qid=str(row.get("id") or row.get("qid") or len(items)),
                question=row["question"],
                relevant_chunks=row.get("relevant_chunk_ids") or row.get("relevant_chunks") or [],
                relevant_docs=row.get("relevant_doc_paths") or row.get("relevant_docs") or []
            ))
    return items

def main(eval_path: str, mode: str="hybrid", alpha: float=0.65, top_k: int=10, mmr: bool=True):
    retr = Retriever(INDEX_PATH, CHUNKS_PATH, EMBED_MODEL)
    items = load_items(eval_path)
    rows = []
    for it in tqdm(items, desc="Retrieval eval"):
        hits = retr.search(it.question, top_k=top_k, mode=mode, mmr=mmr, alpha=alpha)
        got_chunks = [h["chunk_id"] for h in hits]
        got_docs = [h["doc_path"] for h in hits]
        rel_chunks, rel_docs = set(it.relevant_chunks or []), set(it.relevant_docs or [])
        relevant = rel_chunks if rel_chunks else rel_docs
        retrieved = got_chunks if rel_chunks else got_docs
        row = {"qid": it.qid, "mode": mode, "alpha": alpha}
        for k in [1,3,5,10]:
            row[f"P@{k}"] = precision_at_k(relevant, retrieved, k)
            row[f"R@{k}"] = recall_at_k(relevant, retrieved, k)
            row[f"MRR@{k}"] = mrr_at_k(relevant, retrieved, k)
            row[f"nDCG@{k}"] = ndcg_at_k_from_binary(relevant, retrieved, k)
        row["AP@K"] = average_precision(relevant, retrieved, k=top_k)
        rows.append(row)
    import json as _json
    df = pd.DataFrame(rows)
    out = Path("eval_out"); out.mkdir(exist_ok=True)
    df.to_csv(out / "retriever_metrics.csv", index=False)
    agg = df.drop(columns=["qid","mode","alpha"]).mean(numeric_only=True).to_dict()
    (out / "retriever_metrics_summary.json").write_text(_json.dumps({"mode": mode,"alpha":alpha,"top_k":top_k,"mmr":mmr,"aggregate":agg}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved to eval_out/")
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--eval_path", default="data/eval/qa.jsonl")
    p.add_argument("--mode", default="hybrid", choices=["vector","bm25","hybrid"])
    p.add_argument("--alpha", type=float, default=0.65)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--mmr", action="store_true")
    args = p.parse_args()
    main(args.eval_path, args.mode, args.alpha, args.top_k, args.mmr)
