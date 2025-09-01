import os, json, sys, time
from pathlib import Path
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.retriever import Retriever
from src.rag import answer_with_citations
from tools.eval.metrics import exact_match, token_f1, context_precision, context_recall

load_dotenv()
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
INDEX_PATH = os.getenv("INDEX_PATH", "./index/faiss.index")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "./index/chunks.jsonl")

@dataclass
class RagItem:
    qid: str
    question: str
    answers: List[str]
    relevant_chunks: List[str]

def load_items(p: str) -> List[RagItem]:
    items = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            items.append(RagItem(
                qid=str(row.get("id") or row.get("qid") or len(items)),
                question=row["question"],
                answers=row.get("answers") or row.get("gold_answers") or row.get("reference_answers") or [],
                relevant_chunks=row.get("relevant_chunk_ids") or []
            ))
    return items

def main(eval_path: str, mode: str="hybrid", alpha: float=0.65, top_k: int=6, mmr: bool=True):
    retr = Retriever(INDEX_PATH, CHUNKS_PATH, EMBED_MODEL)
    items = load_items(eval_path)
    rows = []
    t0 = time.time()
    for it in tqdm(items, desc="RAG eval"):
        hits = retr.search(it.question, top_k=top_k, mode=mode, mmr=mmr, alpha=alpha)
        context_text = "\n\n---\n\n".join([h["text"] for h in hits])
        out = answer_with_citations(it.question, hits)
        answer = out.get("answer","")
        em  = exact_match(answer, it.answers) if it.answers else 0.0
        f1  = token_f1(answer, it.answers) if it.answers else 0.0
        cp  = context_precision(answer, context_text)
        cr  = context_recall(answer, context_text)
        rows.append({"qid":it.qid,"mode":mode,"alpha":alpha,"EM":em,"F1":f1,"ContextPrecision":cp,"ContextRecall":cr,"answer":answer[:500]})
    import json as _json
    df = pd.DataFrame(rows)
    out = Path("eval_out"); out.mkdir(exist_ok=True)
    df.to_csv(out / "rag_metrics.csv", index=False)
    agg = df.drop(columns=["qid","mode","alpha","answer"]).mean(numeric_only=True).to_dict()
    agg["count"] = len(df); agg["elapsed_sec"] = time.time()-t0
    (out / "rag_metrics_summary.json").write_text(_json.dumps({"mode":mode,"alpha":alpha,"top_k":top_k,"mmr":mmr,"aggregate":agg}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved to eval_out/")
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--eval_path", default="data/eval/qa.jsonl")
    p.add_argument("--mode", default="hybrid", choices=["vector","bm25","hybrid"])
    p.add_argument("--alpha", type=float, default=0.65)
    p.add_argument("--top_k", type=int, default=6)
    p.add_argument("--mmr", action="store_true")
    args = p.parse_args()
    main(args.eval_path, args.mode, args.alpha, args.top_k, args.mmr)
