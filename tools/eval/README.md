# Evaluation Guide

Run:
```bash
python tools/eval/eval_retriever.py --eval_path data/eval/qa.jsonl --mode hybrid --alpha 0.65 --top_k 10 --mmr
python tools/eval/eval_rag.py        --eval_path data/eval/qa.jsonl --mode hybrid --alpha 0.65 --top_k 6  --mmr
```
Outputs are saved to `eval_out/`.
