import json
import math
from typing import List

# qa.jsonl format:
# {"question": "...", "relevant_doc_ids": ["path/to/file1.pdf", "..."]}
# run.jsonl format:
# {"question": "...", "ranking": ["docA", "docB", ...]}

def dcg(rel: List[int]) -> float:
    return sum((r / math.log2(i+2)) for i, r in enumerate(rel))

def ndcg_at_k(rels: List[int], k: int = 5) -> float:
    rels_k = rels[:k]
    ideal = sorted(rels, reverse=True)[:k]
    denom = dcg(ideal) or 1.0
    return dcg(rels_k) / denom

def mrr_at_k(rels: List[int], k: int = 10) -> float:
    for i, r in enumerate(rels[:k]):
        if r > 0:
            return 1.0 / (i+1)
    return 0.0

def evaluate(run_path: str = "run.jsonl", qa_path: str = "data/eval/qa.jsonl", k: int = 10):
    runs = [json.loads(l) for l in open(run_path, "r", encoding="utf-8").read().splitlines()]
    qas = {}
    for l in open(qa_path, "r", encoding="utf-8").read().splitlines():
        j = json.loads(l)
        qas[j["question"]] = set(j["relevant_doc_ids"])

    mrr, ndcg5, rec10 = [], [], []
    for r in runs:
        q = r["question"]
        ranking = r["ranking"]
        gt = qas.get(q, set())
        rels = [1 if (doc in gt) else 0 for doc in ranking]
        mrr.append(mrr_at_k(rels, k))
        ndcg5.append(ndcg_at_k(rels, 5))
        rec10.append(1.0 if any(rels[:10]) else 0.0)

    def avg(x): return sum(x)/max(len(x),1)

    print({
        "MRR@10": round(avg(mrr), 4),
        "nDCG@5": round(avg(ndcg5), 4),
        "Recall@10": round(avg(rec10), 4)
    })

if __name__ == "__main__":
    evaluate()
