#!/usr/bin/env python3
"""Regenerate manuscript figures from GP-only embedding analysis."""
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

matplotlib.use("agg")

BASE = Path(__file__).resolve().parents[2]
RESULTS = BASE / "results" / "gp_revision"
ASSETS = BASE / "manuscript" / "assets" / "refined3_1" / "media"
ASSETS.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────
emb = torch.load(RESULTS / "gp_embeddings.pt", weights_only=False)
embeddings = emb["embeddings"].numpy()
ids = emb["ids"]

df_meta = pd.read_csv(BASE / "data" / "cleaned" / "cleaned_sequences_gp_only.csv")
id_to_virus = dict(zip(df_meta["id"].astype(str), df_meta["virus"]))
labels = [id_to_virus.get(str(i), "Unknown") for i in ids]

virus_colors = {"Ebola": "#c62828", "Lassa": "#1565c0"}
colors = [virus_colors.get(l, "#999") for l in labels]

# ── Figure 1: PCA scatter ───────────────────────────────────────────
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(embeddings)

fig, ax = plt.subplots(figsize=(8, 6))
for virus in ["Ebola", "Lassa"]:
    mask = np.array(labels) == virus
    ax.scatter(
        pca_coords[mask, 0],
        pca_coords[mask, 1],
        c=virus_colors[virus],
        label=virus,
        alpha=0.5,
        s=15,
        edgecolors="none",
    )

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
ax.set_title("ESM-2 Embedding PCA — GP Sequences (n=2,499)", fontsize=13, fontweight="bold")
ax.legend(title="Virus", loc="upper right", frameon=True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(ASSETS / "image1.png", dpi=300, bbox_inches="tight")
fig.savefig(ASSETS / "image1.svg", bbox_inches="tight")
plt.close(fig)
print("Saved image1.png (PCA scatter)")

# ── Figure 2: PCA variance bar ──────────────────────────────────────
# Refit PCA with more components for variance plot
pca_full = PCA(n_components=10)
pca_full.fit(embeddings)
fig, ax = plt.subplots(figsize=(6, 4))
var_explained = pca_full.explained_variance_ratio_[:10] * 100
bars = ax.bar(range(1, 11), var_explained, color="#455a64", edgecolor="white")
ax.set_xlabel("Principal Component", fontsize=12)
ax.set_ylabel("Variance Explained (%)", fontsize=12)
ax.set_title("PCA Variance Explained — GP Embeddings", fontsize=13, fontweight="bold")
for bar, val in zip(bars, var_explained):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(ASSETS / "image2.png", dpi=300, bbox_inches="tight")
fig.savefig(ASSETS / "image2.svg", bbox_inches="tight")
plt.close(fig)
print("Saved image2.png (PCA variance)")

# ── Figure 3: Centroid distance distributions ─────────────────────────
# Compute centroids
ebo_mask = np.array(labels) == "Ebola"
las_mask = np.array(labels) == "Lassa"
ebo_emb = embeddings[ebo_mask]
las_emb = embeddings[las_mask]

ebo_centroid = ebo_emb.mean(axis=0)
las_centroid = las_emb.mean(axis=0)

ebo_dists = np.linalg.norm(ebo_emb - ebo_centroid, axis=1)
las_dists = np.linalg.norm(las_emb - las_centroid, axis=1)

fig, ax = plt.subplots(figsize=(7, 5))
bins = np.linspace(0, max(ebo_dists.max(), las_dists.max()), 60)
ax.hist(ebo_dists, bins=bins, color="#c62828", alpha=0.6, label="Ebola", edgecolor="white")
ax.hist(las_dists, bins=bins, color="#1565c0", alpha=0.6, label="Lassa", edgecolor="white")
ax.axvline(ebo_dists.mean(), color="#c62828", linestyle="--", linewidth=2, label=f"Ebola mean={ebo_dists.mean():.2f}")
ax.axvline(las_dists.mean(), color="#1565c0", linestyle="--", linewidth=2, label=f"Lassa mean={las_dists.mean():.2f}")
ax.set_xlabel("Distance to Class Centroid", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Centroid Distance Distributions — GP Embeddings", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", frameon=True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(ASSETS / "image3.png", dpi=300, bbox_inches="tight")
fig.savefig(ASSETS / "image3.svg", bbox_inches="tight")
plt.close(fig)
print("Saved image3.png (centroid distances)")

# ── Figure 4: Atypicality index distribution ────────────────────────
# Compute per-sequence atypicality
all_dists = np.linalg.norm(embeddings - ebo_centroid, axis=1)
all_dists_las = np.linalg.norm(embeddings - las_centroid, axis=1)

# Use distance to own centroid
atyp_scores = np.where(ebo_mask, all_dists, all_dists_las)

# Z-score within each class
ebo_z = (ebo_dists - ebo_dists.mean()) / ebo_dists.std()
las_z = (las_dists - las_dists.mean()) / las_dists.std()

atyp_index = np.zeros(len(labels))
atyp_index[ebo_mask] = 50 + 15 * ebo_z
atyp_index[las_mask] = 50 + 15 * las_z
atyp_index = np.clip(atyp_index, 0, 100)

fig, ax = plt.subplots(figsize=(7, 5))
bins = np.linspace(0, 100, 50)
ax.hist(atyp_index[ebo_mask], bins=bins, color="#c62828", alpha=0.6, label="Ebola", edgecolor="white")
ax.hist(atyp_index[las_mask], bins=bins, color="#1565c0", alpha=0.6, label="Lassa", edgecolor="white")
ax.axvline(80, color="black", linestyle="--", linewidth=2, label="High-atypicality threshold (80)")
ax.set_xlabel("Atypicality Index (0-100)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Atypicality Index Distribution — GP Embeddings", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", frameon=True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(ASSETS / "image4.png", dpi=300, bbox_inches="tight")
fig.savefig(ASSETS / "image4.svg", bbox_inches="tight")
plt.close(fig)
print("Saved image4.png (atypicality distribution)")

# ── Figure 5: Outlier counts bar chart ──────────────────────────────
high_ebo = int((atyp_index[ebo_mask] >= 80).sum())
high_las = int((atyp_index[las_mask] >= 80).sum())

fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(["Ebola", "Lassa"], [high_ebo, high_las], color=["#c62828", "#1565c0"], edgecolor="white", linewidth=2)
for bar, val in zip(bars, [high_ebo, high_las]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            str(val), ha="center", va="bottom", fontsize=14, fontweight="bold")
ax.set_ylabel("High-Atypicality Count (≥80)", fontsize=12)
ax.set_title("High-Atypicality Outliers — GP Embeddings", fontsize=13, fontweight="bold")
ax.set_ylim(0, max(high_ebo, high_las) * 1.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(ASSETS / "image5.png", dpi=300, bbox_inches="tight")
fig.savefig(ASSETS / "image5.svg", bbox_inches="tight")
plt.close(fig)
print(f"Saved image5.png (outlier counts: EBOV={high_ebo}, LASV={high_las})")

# ── Figure 6: Separation ratio visualization ────────────────────────
# Distance between centroids / pooled std
centroid_dist = np.linalg.norm(ebo_centroid - las_centroid)
pooled_std = np.sqrt((ebo_dists.var() + las_dists.var()) / 2)
sep_ratio = centroid_dist / pooled_std

fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(["Centroid Distance", "Pooled Std"], [centroid_dist, pooled_std], color=["#2e7d32", "#fbc02d"], edgecolor="white", linewidth=2)
ax.text(centroid_dist / 2, 0, f"Distance = {centroid_dist:.2f}", ha="center", va="center", fontsize=12, fontweight="bold", color="white")
ax.text(pooled_std / 2, 1, f"Pooled Std = {pooled_std:.2f}", ha="center", va="center", fontsize=12, fontweight="bold", color="black")
ax.set_title(f"Separation Ratio = {sep_ratio:.2f}", fontsize=13, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(ASSETS / "image6.png", dpi=300, bbox_inches="tight")
fig.savefig(ASSETS / "image6.svg", bbox_inches="tight")
plt.close(fig)
print("Saved image6.png (separation ratio)")

# ── Figure 7: Feature importance or composition ─────────────────────
# Show amino-acid composition comparison between EBOV and LASV GP
aa_list = list("ACDEFGHIKLMNPQRSTVWY")

# Compute mean composition from features.csv if available
features_csv = RESULTS / "features.csv"
if features_csv.exists():
    feat_df = pd.read_csv(features_csv)
    ebo_comp = feat_df[feat_df["virus"] == "Ebola"][[c for c in feat_df.columns if c.startswith("aa_")]].mean()
    las_comp = feat_df[feat_df["virus"] == "Lassa"][[c for c in feat_df.columns if c.startswith("aa_")]].mean()
    aa_ebo = [ebo_comp.get(f"aa_{aa}", 0) for aa in aa_list]
    aa_las = [las_comp.get(f"aa_{aa}", 0) for aa in aa_list]
else:
    # Fallback: compute from sequences
    ebo_seqs = df_meta[df_meta["virus"] == "Ebola"]["sequence"].tolist()
    las_seqs = df_meta[df_meta["virus"] == "Lassa"]["sequence"].tolist()
    aa_ebo = [sum(s.count(aa) for s in ebo_seqs) / sum(len(s) for s in ebo_seqs) for aa in aa_list]
    aa_las = [sum(s.count(aa) for s in las_seqs) / sum(len(s) for s in las_seqs) for aa in aa_list]

x = np.arange(len(aa_list))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width / 2, aa_ebo, width, label="Ebola", color="#c62828", alpha=0.8, edgecolor="white")
ax.bar(x + width / 2, aa_las, width, label="Lassa", color="#1565c0", alpha=0.8, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(aa_list)
ax.set_xlabel("Amino Acid", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_title("Mean Amino-Acid Composition — GP Sequences", fontsize=13, fontweight="bold")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(ASSETS / "image7.png", dpi=300, bbox_inches="tight")
fig.savefig(ASSETS / "image7.svg", bbox_inches="tight")
plt.close(fig)
print("Saved image7.png (amino-acid composition)")

print("\n=== All 7 figures regenerated ===")
print(f"Location: {ASSETS}")
