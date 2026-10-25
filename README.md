# COSC 4371 Project #15 (variant): Prompt-Injection Detection (Safe-Guard)

Patrick Callaghan, Fall 2026. Midsemester checkpoint.

## Run

```bash
python -m pip install -r requirements.txt
python checkpoint.py
python check_checkpoint.py --split split.csv --scores scores.csv --threshold 0.135581
```
Threshold is lowest with a validation false-positive rate <= 0.05 (5%)

`check_checkpoint.py` is the course checker from Canvas

The data downloads from Hugging Face (`xTRam1/safe-guard-prompt-injection`) when the script runs.

Random seed: 42. Runs in under a minute on a CPU; no GPU needed. Tested with Python 3.13.

`checkpoint.py` writes `split.csv`, `scores.csv` and `roc.png`, and prints the audit,
removal counts, split counts, leakage check, threshold, training time and package versions.