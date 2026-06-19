"""Analyze ESM-2 GP embeddings: centroid separation, outliers, PCA."""
import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

base = "/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model"

# Load embeddings
data = torch.load(f"{base}/results/gp_revision/gp_embeddings.pt", weights_only=False)
ids = data["ids"]
embeddings = data["embeddings"].numpy()

# Load metadata
df = pd.read_csv(f"{base}/data/cleaned/cleaned_sequences_gp_only.csv")
id_to_label = dict(zip(df["id"], df["virus"]))
labels = [id_to_label.get(i, "unknown") for i in ids]

# Separate by virus (CSV uses 'Ebola'/'Lassa')
ebov_mask = np.array([l == "Ebola" for l in labels])
lasv_mask = np.array([l == "Lassa" for l in labels])

ebo_emb = embeddings[ebov_mask]
las_emb = embeddings[lasv_mask]

print(f"EBOV: {ebo_emb.shape[0]} sequences")
print(f"LASV: {las_emb.shape[0]} sequences")

# Centroids
ebo_centroid = ebo_emb.mean(axis=0)
las_centroid = las_emb.mean(axis=0)

# Centroid distance
centroid_dist = np.linalg.norm(ebo_centroid - las_centroid)
print(f"\nCentroid separation (L2): {centroid_dist:.4f}")

# Intra-class spread (mean distance to own centroid)
ebo_spread = np.mean([np.linalg.norm(v - ebo_centroid) for v in ebo_emb])
las_spread = np.mean([np.linalg.norm(v - las_centroid) for v in las_emb])
print(f"EBOV mean spread: {ebo_spread:.4f}")
print(f"LASV mean spread: {las_spread:.4f}")

# Separation ratio
d1 = np.mean([np.linalg.norm(v - ebo_centroid) for v in las_emb])
d2 = np.mean([np.linalg.norm(v - las_centroid) for v in ebo_emb])
sep_ratio = (d1 + d2) / 2 / max(ebo_spread, las_spread)
print(f"Separation ratio: {sep_ratio:.2f}")

# Outliers (>80th percentile distance from own centroid)
ebo_dists = np.array([np.linalg.norm(v - ebo_centroid) for v in ebo_emb])
las_dists = np.array([np.linalg.norm(v - las_centroid) for v in las_emb])

ebo_threshold = np.percentile(ebo_dists, 80)
las_threshold = np.percentile(las_dists, 80)

ebo_outliers = np.sum(ebo_dists > ebo_threshold)
las_outliers = np.sum(las_dists > las_threshold)

print(f"\nEBOV outliers (>80th pct): {ebo_outliers} / {len(ebo_dists)} ({ebo_outliers/len(ebo_dists)*100:.1f}%)")
print(f"LASV outliers (>80th pct): {las_outliers} / {len(las_dists)} ({las_outliers/len(las_dists)*100:.1f}%)")

# Cross-class distances (out-of-distribution flag)
ebo_to_las = np.array([np.linalg.norm(v - las_centroid) for v in ebo_emb])
las_to_ebo = np.array([np.linalg.norm(v - ebo_centroid) for v in las_emb])

# Atypicality scores (z-scores of centroid distance)
ebo_z = (ebo_dists - ebo_dists.mean()) / ebo_dists.std()
las_z = (las_dists - las_dists.mean()) / las_dists.std()

ebo_atypical = np.clip(50 + 15 * ebo_z, 0, 100)
las_atypical = np.clip(50 + 15 * las_z, 0, 100)

print(f"\nEBOV atypicality: mean={ebo_atypical.mean():.1f}, std={ebo_atypical.std():.1f}")
print(f"LASV atypicality: mean={las_atypical.mean():.1f}, std={las_atypical.std():.1f}")

# High atypicality (>=80)
ebo_high = np.sum(ebo_atypical >= 80)
las_high = np.sum(las_atypical >= 80)
print(f"EBOV high atypicality (>=80): {ebo_high}")
print(f"LASV high atypicality (>=80): {las_high}")

# PCA for visualization
scaler = StandardScaler()
emb_scaled = scaler.fit_transform(embeddings)
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(emb_scaled)

print(f"\nPCA variance explained: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%")

# Save results
results = {
    "centroid_dist": float(centroid_dist),
    "ebo_spread": float(ebo_spread),
    "las_spread": float(las_spread),
    "sep_ratio": float(sep_ratio),
    "ebo_outliers": int(ebo_outliers),
    "las_outliers": int(las_outliers),
    "ebo_outlier_pct": float(ebo_outliers/len(ebo_dists)*100),
    "las_outlier_pct": float(las_outliers/len(las_dists)*100),
    "ebo_high_atypical": int(ebo_high),
    "las_high_atypical": int(las_high),
    "pca_var_pc1": float(pca.explained_variance_ratio_[0]*100),
    "pca_var_pc2": float(pca.explained_variance_ratio_[1]*100),
}

import json
with open(f"{base}/results/gp_revision/esm2_analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {base}/results/gp_revision/esm2_analysis_results.json")
