import math
import re
from collections import Counter
from typing import List

_WORD = re.compile(r"\w+", re.UNICODE)

def tokenize(text: str) -> List[str]:
    return [w for w in _WORD.findall(text.lower()) if len(w) >= 2]

class BM25Okapi:
    """Simple BM25 implementation over in-memory list-of-tokens corpus."""
    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus_tokens
        self.N = len(corpus_tokens)
        self.doc_freq = Counter()
        self.doc_len = [len(doc) for doc in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N > 0 else 0.0
        self.tf = [Counter(doc) for doc in corpus_tokens]
        # document frequencies
        for doc in corpus_tokens:
            for term in set(doc):
                self.doc_freq[term] += 1
        # precompute idf
        self.idf = {}
        for term, df in self.doc_freq.items():
            # idf with +0.5 smoothing
            self.idf[term] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: List[str], idx: int) -> float:
        score = 0.0
        if idx >= self.N:
            return score
        dl = self.doc_len[idx] or 1
        tf = self.tf[idx]
        for term in query_tokens:
            if term not in tf:
                continue
            idf = self.idf.get(term, 0.0)
            freq = tf[term]
            denom = freq + self.k1 * (1.0 - self.b + self.b * dl / (self.avgdl or 1.0))
            score += idf * (freq * (self.k1 + 1.0) / (denom or 1.0))
        return score

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        return [self.score(query_tokens, i) for i in range(self.N)]
