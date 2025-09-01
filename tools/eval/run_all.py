import os, sys, subprocess
from pathlib import Path
root = Path(__file__).resolve().parents[2]
eval_seed = root / "data/eval/questions_seed.jsonl"
qa_path = root / "data/eval/qa.jsonl"
def run(cmd): print("\n$", " ".join(cmd)); subprocess.run(cmd, check=False)
if __name__ == "__main__":
    run([sys.executable, "tools/eval/autolabel_build_eval.py", "--seed_path", str(eval_seed), "--out_path", str(qa_path)])
    run([sys.executable, "tools/eval/eval_retriever.py", "--eval_path", str(qa_path), "--mode", "hybrid", "--alpha", "0.65", "--top_k", "10", "--mmr"])
    run([sys.executable, "tools/eval/eval_rag.py", "--eval_path", str(qa_path), "--mode", "hybrid", "--alpha", "0.65", "--top_k", "6", "--mmr"])
    print("Done. Check eval_out/")
