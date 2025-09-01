import math, re
from typing import Sequence, Set

def dcg(relevances: Sequence[float], k: int | None = None) -> float:
    if k is None: k = len(relevances)
    return sum((rel / math.log2(i + 2)) for i, rel in enumerate(relevances[:k]))

def ndcg(relevances: Sequence[float], ideal_relevances: Sequence[float], k: int | None = None) -> float:
    if k is None: k = len(relevances)
    denom = dcg(ideal_relevances, k)
    return 0.0 if denom <= 0 else dcg(relevances, k) / denom

def precision_at_k(relevant: Set[str], retrieved: Sequence[str], k: int) -> float:
    if k <= 0: return 0.0
    cut = retrieved[:k]
    return sum(1 for x in cut if x in relevant) / k

def recall_at_k(relevant: Set[str], retrieved: Sequence[str], k: int) -> float:
    if not relevant: return 0.0
    cut = retrieved[:k]
    return sum(1 for x in cut if x in relevant) / len(relevant)

def mrr_at_k(relevant: Set[str], retrieved: Sequence[str], k: int) -> float:
    for i, x in enumerate(retrieved[:k], start=1):
        if x in relevant: return 1.0 / i
    return 0.0

def average_precision(relevant: Set[str], retrieved: Sequence[str], k: int | None = None) -> float:
    if k is None: k = len(retrieved)
    num_hits = 0; score = 0.0
    for i, x in enumerate(retrieved[:k], start=1):
        if x in relevant:
            num_hits += 1
            score += num_hits / i
    return 0.0 if not relevant else score / min(len(relevant), k)

def ndcg_at_k_from_binary(relevant: Set[str], retrieved: Sequence[str], k: int) -> float:
    rels = [1.0 if x in relevant else 0.0 for x in retrieved[:k]]
    ideal = sorted(rels, reverse=True)
    return ndcg(rels, ideal, k=k)

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())

def _tokens(s: str) -> list[str]:
    return re.findall(r"\w+", _normalize(s), flags=re.UNICODE)

def exact_match(pred: str, gold_list: Sequence[str]) -> float:
    p = _normalize(pred)
    return 1.0 if any(p == _normalize(g) for g in gold_list) else 0.0

def token_f1(pred: str, gold_list: Sequence[str]) -> float:
    best = 0.0; ptoks = _tokens(pred)
    from collections import Counter
    for g in gold_list:
        gtoks = _tokens(g)
        if not ptoks and not gtoks: return 1.0
        if not ptoks or not gtoks: continue
        cp, cg = Counter(ptoks), Counter(gtoks)
        overlap = sum(min(cp[t], cg[t]) for t in cp if t in cg)
        if overlap == 0: f1 = 0.0
        else:
            prec = overlap / len(ptoks); rec = overlap / len(gtoks)
            f1 = 0.0 if prec + rec == 0 else 2*prec*rec/(prec+rec)
        best = max(best, f1)
    return best

def context_precision(answer: str, context: str) -> float:
    at, ct = _tokens(answer), set(_tokens(context))
    return 0.0 if not at else sum(1 for t in at if t in ct) / len(at)

def context_recall(answer: str, context: str) -> float:
    at, ct = set(_tokens(answer)), _tokens(context)
    return 0.0 if not ct else sum(1 for t in ct if t in at) / len(ct)
