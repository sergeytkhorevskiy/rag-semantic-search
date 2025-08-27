
import os, json, requests, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=os.environ.get("API_URL","http://127.0.0.1:8000"), help="Base URL of FastAPI (without trailing slash)")
    ap.add_argument("--qa", default="data/eval/qa_min.jsonl", help="Path to qa jsonl")
    ap.add_argument("--top_k", type=int, default=int(os.environ.get("TOP_K","6")))
    ap.add_argument("--out", default="run_min.jsonl", help="Output run (basenames)")
    ap.add_argument("--out_full", default="run_min_fullpaths.jsonl", help="Output run (full paths)")
    args = ap.parse_args()

    with open(args.qa, "r", encoding="utf-8") as f:
        qas = [json.loads(l) for l in f if l.strip()]

    out_bn, out_full = [], []
    for item in qas:
        q = item["question"]
        r = requests.get(f"{args.api}/search", params={"q": q, "top_k": args.top_k}, timeout=60)
        r.raise_for_status()
        hits = r.json()["hits"]
        ranking_full = [h["doc_path"] for h in hits]
        ranking_bn = [os.path.basename(p) for p in ranking_full]
        out_full.append({"question": q, "ranking": ranking_full})
        out_bn.append({"question": q, "ranking": ranking_bn})
        print(f"[OK] {q} -> {ranking_bn}")

    with open(args.out_full, "w", encoding="utf-8") as f:
        for row in out_full:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(args.out, "w", encoding="utf-8") as f:
        for row in out_bn:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
