# Автогенерація eval-набору (silver labels)
1) Побудуйте індекс: `python scripts/ingest.py`
2) Перевірте/оновіть `data/eval/questions_seed.jsonl`
3) Згенеруйте `qa.jsonl`: `python tools/eval/autolabel_build_eval.py --seed_path data/eval/questions_seed.jsonl --out_path data/eval/qa.jsonl`
4) Запустіть оцінку: `python tools/eval/run_all.py`
