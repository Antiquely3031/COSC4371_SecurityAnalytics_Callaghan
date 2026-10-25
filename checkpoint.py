"""
Created by Patrick Callaghan (2523109)

COSC4371 Security Analytics - Midsemester Checkpoint
Project: #15 Variant (Safe-Guard only)
"""

import re
import sys
import time
import unicodedata

import numpy as np
import pandas as pd
import sklearn
from datasets import load_dataset
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

SEED = 42
COS_DUP_SIMILARITY = 0.90
TARGET_FPR = 0.05

# Load Safe-Guard Dataset
ds = load_dataset("xTRam1/safe-guard-prompt-injection")
parts = []
for split in ds:
    df = ds[split].to_pandas()
    df["orig_split"] = split
    df["row_id"] = [f"Safe-Guard:{split}:{i}" for i in range(len(df))]
    parts.append(df)
data = pd.concat(parts, ignore_index=True)

# Audited Data
print("\n== AUDIT ==")
print("Rows, Columns:", len(data), list(data.columns[:2]))
print("Class Counts:", data["label"].value_counts().to_dict())
print("Class Percentage:", (100 * data["label"].value_counts(normalize=True)).round(1).to_dict())
print("Missing Label(s):", data["label"].isna().sum(), "| Missing Text(s): ", data["text"].isna().sum())

# Find Identical 'Text' String(s)
counts = data["text"].value_counts()
duplicates = counts[counts > 1]
print(f"Exact Duplicates: {duplicates.sum()} rows in {len(duplicates)} groups ({duplicates.sum() - len(duplicates)} extra copies)")

# Find Near Duplicates (where cosine similarity >= 0.90 for 3-5-gram TF-IDF vectors,
# post NFKC normalization, lowercasing and collapsing whitespace
def clean_text(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).lower()).strip()

# Vectorize
X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(data["text"].map(clean_text))
rows, cols = [], []
for start in range(0, X.shape[0], 500):
    sims = (X[start:start + 500] @ X.T).toarray()
    r, c = np.nonzero(sims >= COS_DUP_SIMILARITY)
    rows.append(r + start)
    cols.append(c)
graph = coo_matrix((np.ones(sum(map(len, rows))), (np.concatenate(rows), np.concatenate(cols))), shape=(len(data), len(data)))
_, data["group"] = connected_components(graph, directed=False)
gsize = data["group"].map(data["group"].value_counts())
print(f"Near Duplicates (includes exact duplicates): {(gsize > 1).sum()} rows in {data.loc[gsize > 1, 'group'].nunique()} groups")

# Removed Data
print("\n== REMOVALS ==")
n0 = len(data)

# Keep one copy of exact duplicates, prefer copy from publisher's test split
data = data.sort_values("orig_split", kind="stable")
data = data.drop_duplicates("text", keep="first")
print("Number of Copies of Exact Duplicates that were Removed:", n0 - len(data))

# In near duplicates, drop train rows whose group also has a test row
group_has_test = data.groupby("group")["orig_split"].transform(lambda s: (s == "test").any())
drop = group_has_test & (data["orig_split"] == "train")
data = data[~drop].copy()
print("Near-Duplicate Train Rows Removed:", drop.sum())

# Validation: take 1 of 7 folds of train, whole groups only, balanced labels
data["split"] = data["orig_split"]
train_idx = data.index[data["split"] == "train"]
folds = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
_, val_pos = next(folds.split(train_idx, data.loc[train_idx, "label"], data.loc[train_idx, "group"]))
data.loc[train_idx[val_pos], "split"] = "validation"
print(data.groupby(["split", "label"]).size().unstack())

# Check Leakage
train_text = data.loc[data["split"] == "train", "text"]
leak_test = train_text.isin(data.loc[data["split"] == "test", "text"]).sum()
leak_validation = train_text.isin(data.loc[data["split"] == "validation", "text"]).sum()
print(f"Leakage (train rows also in test): {leak_test}") 
print(f"Leakage (train rows also in validation): {leak_validation}")
if leak_test or leak_validation:
    sys.exit("Leakage found; stopping.")
data["dataset"] = "safeguard"
data[["row_id", "dataset", "split", "label"]].to_csv("split.csv", index=False)

# Establish Baseline
train, validation, test = (data[data["split"] == s] for s in ("train", "validation", "test"))
model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),)
t0 = time.time()
model.fit(train["text"], train["label"])
print(f"\n== BASELINE ==\nTraining Time: {time.time() - t0:.2f} s")
print("Vocabulary size:", len(model[0].vocabulary_))

# Threshold: lowest value, set to 6 decimals, where 0.05, at most, of validation benign prompts score >= it
benign = np.sort(model.predict_proba(validation.loc[validation["label"] == 0, "text"])[:, 1])
grid = np.arange(0, 1_000_001) / 1_000_000
fpr_at = (len(benign) - np.searchsorted(benign, grid)) / len(benign)
threshold = grid[np.argmax(fpr_at <= TARGET_FPR)]
print(f"Threshold: {threshold:.6f}")

# Score test set once
test_scores = model.predict_proba(test["text"])[:, 1]
pd.DataFrame({"row_id": test["row_id"], "test_set": "safeguard_clean", "label" : test["label"], "score": test_scores}).to_csv("scores.csv", index=False)

# Print Plotted Figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(test["label"], test_scores)
pct = 100 * test["label"].mean()
plt.figure(figsize=(6, 4.5))
plt.plot(fpr, tpr, lw=2, label="TF-IDF + Logistic Regression")
xs = np.logspace(-4, 0, 200)
plt.plot(xs, xs, "k--", lw=1, label="Random Scorer")
plt.xscale("log")
plt.xlim(1 / (test["label"] == 0).sum() / 2, 1)
plt.xlabel("False-Positive Rate (log scale)")
plt.ylabel("True-Positive Rate (recall)")
plt.title(f"Safe-Guard Test Set: {len(test)} rows, {pct:.1f}% injections")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc.png", dpi=200)

print(f"\nPython {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}, "f"scikit-learn {sklearn.__version__}")
