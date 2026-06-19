"""Full re-analysis pipeline for GP-only dataset."""
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys

base = "/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model"

# Load cleaned GP-only data
df = pd.read_csv(f"{base}/data/cleaned/cleaned_sequences_gp_only.csv")
print("=== GP-ONLY DATASET ===")
print(df.groupby("virus")["length"].agg(["count", "mean", "min", "max"]).round(1))

# Filter out extreme short sequences (likely errors)
df = df[df["length"] >= 300].copy()
print(f"\nAfter filtering length >= 300: {len(df)} sequences")
print(df.groupby("virus")["length"].agg(["count", "mean", "min", "max"]).round(1))

# ---- 1. COMPOSITION FEATURES ----
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

def aa_freq_features(sequences):
    rows = []
    for seq in sequences:
        length = len(seq)
        row = {"seq_length": length}
        for aa in AMINO_ACIDS:
            row[f"aa_freq_{aa}"] = seq.count(aa) / length
        rows.append(row)
    return pd.DataFrame(rows)

features = aa_freq_features(df["sequence"].tolist())
features["virus"] = df["virus"].values
features["id"] = df["id"].values

# Save features
os.makedirs(f"{base}/results/gp_revision", exist_ok=True)
features.to_csv(f"{base}/results/gp_revision/features.csv", index=False)

# ---- 2. SITE-LEVEL CONSERVATION (simplified) ----
# For each virus, compute per-position entropy
# This is a simplified version - full alignment would be better

def compute_position_entropy(sequences, max_pos=700):
    entropy = []
    for pos in range(max_pos):
        chars = [seq[pos] for seq in sequences if pos < len(seq)]
        if not chars:
            break
        counts = Counter(chars)
        total = len(chars)
        probs = [c/total for c in counts.values()]
        H = -sum(p * np.log2(p) for p in probs if p > 0)
        entropy.append(H)
    return np.array(entropy)

ebola_seqs = df[df["virus"] == "Ebola"]["sequence"].tolist()
lassa_seqs = df[df["virus"] == "Lassa"]["sequence"].tolist()

ebola_ent = compute_position_entropy(ebola_seqs)
lassa_ent = compute_position_entropy(lassa_seqs)

# Classify sites
def classify_sites(entropy):
    categories = []
    for H in entropy:
        if H < 0.1:
            categories.append("Critical")
        elif H < 0.5:
            categories.append("Conserved")
        elif H < 1.0:
            categories.append("Intermediate")
        else:
            categories.append("Hotspot")
    return categories

ebola_cat = classify_sites(ebola_ent)
lassa_cat = classify_sites(lassa_ent)

ebola_counts = Counter(ebola_cat)
lassa_counts = Counter(lassa_cat)

print("\n=== SITE CATEGORIES ===")
print(f"EBOV GP ({len(ebola_seqs)} seqs, {len(ebola_cat)} positions):")
for cat in ["Critical", "Conserved", "Intermediate", "Hotspot"]:
    n = ebola_counts.get(cat, 0)
    print(f"  {cat}: {n} ({n/len(ebola_cat)*100:.1f}%)")

print(f"\nLASV GP ({len(lassa_seqs)} seqs, {len(lassa_cat)} positions):")
for cat in ["Critical", "Conserved", "Intermediate", "Hotspot"]:
    n = lassa_counts.get(cat, 0)
    print(f"  {cat}: {n} ({n/len(lassa_cat)*100:.1f}%)")

# Save
site_df = pd.DataFrame({
    "virus": ["Ebola"] * len(ebola_cat) + ["Lassa"] * len(lassa_cat),
    "position": list(range(len(ebola_cat))) + list(range(len(lassa_cat))),
    "entropy": list(ebola_ent) + list(lassa_ent[:len(lassa_cat)]),
    "category": ebola_cat + lassa_cat
})
site_df.to_csv(f"{base}/results/gp_revision/site_categories.csv", index=False)

# ---- 3. TRAIN CLASSIFIER ----
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

X = features[[c for c in features.columns if c.startswith("aa_freq_") or c == "seq_length"]].values
y = (features["virus"] == "Ebola").astype(int).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_prob),
    "n_train": len(y_train),
    "n_test": len(y_test),
    "n_ebola_train": sum(y_train),
    "n_lassa_train": len(y_train) - sum(y_train),
}

print("\n=== CLASSIFICATION (GP vs GP) ===")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

# Save model and metrics
import joblib
os.makedirs(f"{base}/models/gp_revision", exist_ok=True)
joblib.dump(clf, f"{base}/models/gp_revision/gp_classifier.joblib")
pd.Series(metrics).to_json(f"{base}/models/gp_revision/training_metrics.json")

# ---- 4. FIGURES ----
fig_dir = f"{base}/results/gp_revision/figures"
os.makedirs(fig_dir, exist_ok=True)

# Fig 1: Site category comparison
fig, ax = plt.subplots(figsize=(8, 4))
cats = ["Critical", "Conserved", "Intermediate", "Hotspot"]
ebola_pcts = [ebola_counts.get(c, 0) / len(ebola_cat) * 100 for c in cats]
lassa_pcts = [lassa_counts.get(c, 0) / len(lassa_cat) * 100 for c in cats]

x = np.arange(len(cats))
width = 0.35
ax.bar(x - width/2, ebola_pcts, width, label="EBOV GP", color="#d62728")
ax.bar(x + width/2, lassa_pcts, width, label="LASV GP", color="#1f77b4")
ax.set_ylabel("Fraction (%)")
ax.set_title("Site Category Distribution: GP vs GP")
ax.set_xticks(x)
ax.set_xticklabels(cats)
ax.legend()
plt.tight_layout()
plt.savefig(f"{fig_dir}/site_category_comparison_gp.png", dpi=150)
plt.close()
print(f"Saved: {fig_dir}/site_category_comparison_gp.png")

# Fig 2: Length distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df[df["virus"] == "Ebola"]["length"], bins=30, alpha=0.7, label="EBOV GP", color="#d62728")
ax.hist(df[df["virus"] == "Lassa"]["length"], bins=30, alpha=0.7, label="LASV GP", color="#1f77b4")
ax.set_xlabel("Sequence Length (amino acids)")
ax.set_ylabel("Count")
ax.set_title("GP Sequence Length Distribution")
ax.legend()
plt.tight_layout()
plt.savefig(f"{fig_dir}/length_distribution_gp.png", dpi=150)
plt.close()
print(f"Saved: {fig_dir}/length_distribution_gp.png")

# Fig 3: Entropy profile
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(ebola_ent[:500], label="EBOV GP", color="#d62728", alpha=0.8)
ax.plot(lassa_ent[:500], label="LASV GP", color="#1f77b4", alpha=0.8)
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Hotspot threshold")
ax.set_xlabel("Position")
ax.set_ylabel("Shannon Entropy")
ax.set_title("Per-Position Entropy Profile (GP)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{fig_dir}/entropy_profile_gp.png", dpi=150)
plt.close()
print(f"Saved: {fig_dir}/entropy_profile_gp.png")

print("\n=== ALL DONE ===")
print(f"Results in: {base}/results/gp_revision/")
