
import json, math, os, argparse

def dcg(rel):
    return sum((r / (math.log2(i+2))) for i, r in enumerate(rel))

def ndcg_at_k(rels, k=5):
    rels_k = rels[:k]
    ideal = sorted(rels, reverse=True)[:k]
    denom = dcg(ideal) or 1.0
    return dcg(rels_k) / denom

def mrr_at_k(rels, k=10):
    for i, r in enumerate(rels[:k]):
        if r > 0:
            return 1.0 / (i+1)
    return 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="data/eval/qa_min.jsonl")
    ap.add_argument("--run", default="run_min.jsonl")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    qas = {}
    with open(args.qa, "r", encoding="utf-8") as f:
        for l in f:
            if not l.strip(): continue
            j = json.loads(l)
            # normalize GT to basenames
            qas[j["question"]] = set([os.path.basename(x) for x in j["relevant_doc_ids"]])

    runs = [json.loads(l) for l in open(args.run, "r", encoding="utf-8").read().splitlines()]

    mrr, ndcg5, rec10 = [], [], []
    for r in runs:
        q = r["question"]
        ranking = [os.path.basename(x) for x in r["ranking"]]
        gt = qas.get(q, set())
        rels = [1 if (doc in gt) else 0 for doc in ranking]
        mrr.append(mrr_at_k(rels, args.k))
        ndcg5.append(ndcg_at_k(rels, 5))
        rec10.append(1.0 if any(rels[:10]) else 0.0)

    def avg(x): return sum(x)/max(len(x),1)

    print({
        "MRR@10": round(avg(mrr), 4),
        "nDCG@5": round(avg(ndcg5), 4),
        "Recall@10": round(avg(rec10), 4)
    })

if __name__ == "__main__":
    main()
