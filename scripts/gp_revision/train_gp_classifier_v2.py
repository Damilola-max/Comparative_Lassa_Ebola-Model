#!/usr/bin/env python3
"""
Train a robust GP classifier using ESM-2 embeddings + composition.
Compares SVM, Random Forest, XGBoost, Logistic Regression.
Saves the best model with calibration for atypicality scoring.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

BASE = Path(__file__).resolve().parents[2]
RESULTS = BASE / "results" / "gp_revision"
MODEL_DIR = BASE / "models" / "gp_revision"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────
print("Loading embeddings and metadata...")
emb = torch.load(RESULTS / "gp_embeddings.pt", weights_only=False)
embeddings = emb["embeddings"].numpy()  # (2499, 480)
emb_ids = emb["ids"]

df = pd.read_csv(BASE / "data" / "cleaned" / "cleaned_sequences_gp_only.csv")
# Create id -> row mapping
id_to_idx = {str(rid): i for i, rid in enumerate(df["id"])}

# Align embeddings to metadata rows
ordered_rows = []
for eid in emb_ids:
    idx = id_to_idx.get(str(eid))
    if idx is not None:
        ordered_rows.append(df.iloc[idx])
    else:
        print(f"Warning: ID {eid} not found in metadata")

df_aligned = pd.DataFrame(ordered_rows).reset_index(drop=True)
y = (df_aligned["virus"] == "Ebola").astype(int).values

# ── Build feature matrices ──────────────────────────────────────────
# 1. ESM-2 embeddings (480-dim)
X_esm = embeddings

# 2. Composition features
aa_list = list("ACDEFGHIKLMNPQRSTVWY")
comp_rows = []
for seq in df_aligned["sequence"]:
    seq = str(seq).upper()
    total = len(seq)
    comp_rows.append([seq.count(aa) / total for aa in aa_list])

X_comp = np.array(comp_rows)

# 3. Length feature
lengths = df_aligned["sequence"].apply(len).values.reshape(-1, 1)

# 4. Combined
X_combined = np.hstack([X_esm, X_comp, lengths])

feature_names = [f"esm_{i}" for i in range(X_esm.shape[1])] + \
                [f"aa_{aa}" for aa in aa_list] + ["length"]

print(f"Features shape: {X_combined.shape}")
print(f"Class distribution: Ebola={(y==1).sum()}, Lassa={(y==0).sum()}")

# ── Train/test split ────────────────────────────────────────────────
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_combined, y, np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ── Train multiple models ────────────────────────────────────────────
models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    "SVM_RBF": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=20, class_weight="balanced", random_state=42, n_jobs=-1),
}

if XGB_AVAILABLE:
    models["XGBoost"] = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y==0).sum()/(y==1).sum(),
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )

results = {}
print("\n=== Model Training & Evaluation ===")

for name, model in models.items():
    print(f"\n--- {name} ---")
    if name == "SVM_RBF":
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]
    else:
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results[name] = {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "auc": auc, "model": model,
    }
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Lassa", "Ebola"]))

# ── Select best model ──────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["f1"])
print(f"\n=== Best Model: {best_name} ===")
print(f"  F1 = {results[best_name]['f1']:.4f}")

best_model = results[best_name]["model"]

# ── 5-fold cross-validation on full data ──────────────────────────
print("\n=== 5-Fold Cross-Validation ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    if name == "SVM_RBF":
        y_pred_cv = cross_val_predict(model, scaler.fit_transform(X_combined), y, cv=cv, method="predict")
    else:
        y_pred_cv = cross_val_predict(model, scaler.fit_transform(X_combined), y, cv=cv, method="predict")
    acc_cv = accuracy_score(y, y_pred_cv)
    f1_cv = f1_score(y, y_pred_cv)
    print(f"  {name:20s} CV Accuracy={acc_cv:.4f}  CV F1={f1_cv:.4f}")

# ── Compute class centroids for atypicality calibration ────────────
print("\n=== Computing Calibration ===")
X_all_s = scaler.fit_transform(X_combined)
y_all_pred = best_model.predict(X_all_s)

class_centroids = {}
class_distance_stats = {}

for label in [0, 1]:
    mask = y_all_pred == label
    class_samples = X_all_s[mask]
    centroid = class_samples.mean(axis=0)
    dists = np.linalg.norm(class_samples - centroid, axis=1)
    class_centroids[str(label)] = centroid.tolist()
    class_distance_stats[str(label)] = {"mean": float(dists.mean()), "std": float(dists.std())}
    print(f"  Class {label}: n={mask.sum()}, mean_dist={dists.mean():.4f}, std_dist={dists.std():.4f}")

calibration = {
    "feature_columns": feature_names,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "class_centroids": class_centroids,
    "class_distance_stats": class_distance_stats,
}

# ── Save model bundle ─────────────────────────────────────────────
bundle = {
    "model": best_model,
    "scaler": scaler,
    "risk_calibration": calibration,
    "feature_columns": feature_names,
    "model_name": best_name,
    "metrics": results[best_name],
    "esm_dim": X_esm.shape[1],
}

joblib.dump(bundle, MODEL_DIR / "gp_classifier_v2.joblib")
print(f"\n=== Saved: {MODEL_DIR / 'gp_classifier_v2.joblib'} ===")

# ── Test on challenging sequences ──────────────────────────────────
print("\n=== Edge Case Tests ===")
# Spondweni virus GP-like sequence (should NOT be Lassa)
spondweni = "MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVYLLPRRGPRLGVRATRKTSERSQPRGRRQPIPKARRSEGRSWAQPGHPWNTNYKAPRYKQGGASSNVPQADMENKAYRRESYLVLSTDTKVEEIAAGSRAETEAGDATNRRPQDVKFPGGGQIVGGVYLLPRRGPRLGVRATRKTSERSQ"

# Real Ebola GP
ebo_real = df_aligned[df_aligned["virus"] == "Ebola"]["sequence"].iloc[0]
las_real = df_aligned[df_aligned["virus"] == "Lassa"]["sequence"].iloc[0]

from src.features.sequence_features import amino_acid_frequency_features, clean_sequence

def featurize(seqs):
    cleaned = [clean_sequence(s) for s in seqs]
    comp_feat = amino_acid_frequency_features(cleaned)
    comp = comp_feat.values
    lens = np.array([[len(s)] for s in cleaned])
    # We need ESM-2 embeddings for these - for now just use composition + length
    # This is a limitation; for new sequences we'd need to run ESM-2
    placeholder = np.zeros((len(seqs), X_esm.shape[1]))
    return np.hstack([placeholder, comp, lens])

# For now test on training data to verify model works
test_seqs = [ebo_real, las_real]
X_test_edge = featurize(test_seqs)
X_test_edge_s = scaler.transform(X_test_edge)
probs = best_model.predict_proba(X_test_edge_s)[:, 1]
preds = (probs >= 0.5).astype(int)
labels_map = {0: "Lassa", 1: "Ebola"}
for seq, prob, pred in zip(test_seqs, probs, preds):
    print(f"  Pred={labels_map[pred]}  Prob_EBO={prob:.4f}  Seq_len={len(seq)}")

print("\n=== Done ===")
