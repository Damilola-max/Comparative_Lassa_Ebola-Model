"""
Generate an interactive-style 3D step-by-step pipeline flowchart using matplotlib.
Outputs: PNG and PDF vector graphic.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

OUT_DIR = Path(__file__).resolve().parents[0]
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Pipeline stages: (stage_x, sub_y, depth_z, label, details)
# ---------------------------------------------------------------------------
STAGES = [
    (0, 0, 0, "1. Data Acquisition", "NCBI / EpiFlu\nRaw FASTA"),
    (1, 0, 0, "2. Sequence Cleaning", "Remove X/B/Z\nStrip U/O\nDrop <50 AA"),
    (2, 0, 0, "3. Metadata Parsing", "Accession\nCountry / Date\nLineage"),
    (3, -1, -1, "4A. Composition\nFeatures", "20 AA frequencies\n+ length"),
    (3, 1, 1, "4B. ESM-2 Embeddings", "esm2_t12_35M\n480-dim mean-pool"),
    (4, -1, -1, "5A. Classifier\nTraining", "Random Forest /\nLogistic Regression"),
    (4, 1, 1, "5B. Centroid\nCalibration", "Class centroids\nRisk bands"),
    (5, 0, 0, "6. Validation & QA", "Stratified CV\nGrouped CV\nAblation"),
    (6, 0, 0, "7. Site-Level Analysis", "Shannon entropy\nGP alignment map"),
    (7, 0, 0, "8. Streamlit Dashboard", "Upload FASTA\nReal-time charts\nCSV / PDF export"),
]

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (2, 4),
    (3, 5), (4, 6), (5, 7), (6, 7), (7, 8)
]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(18, 12))
ax = fig.add_subplot(111, projection="3d")

xs, ys, zs = [], [], []
for x, y, z, label, details in STAGES:
    xs.append(x)
    ys.append(y)
    zs.append(z)

# Draw nodes as 3D boxes (using scatter + text)
ax.scatter(xs, ys, zs, s=2000, c="#1976d2", alpha=0.9, edgecolors="black", linewidths=1.5, zorder=5)

for i, (x, y, z, label, details) in enumerate(STAGES):
    ax.text(x, y, z + 0.15, label, fontsize=10, ha="center", va="bottom", fontweight="bold", color="white")
    ax.text(x, y, z - 0.12, details, fontsize=8, ha="center", va="top", color="white", linespacing=1.2)

# Draw connections
for src, dst in CONNECTIONS:
    x1, y1, z1 = xs[src], ys[src], zs[src]
    x2, y2, z2 = xs[dst], ys[dst], zs[dst]
    ax.plot([x1, x2], [y1, y2], [z1, z2], color="#455a64", linewidth=2, alpha=0.7)

ax.set_xlabel("Pipeline Stage", fontsize=12, fontweight="bold")
ax.set_ylabel("Parallel Track", fontsize=12, fontweight="bold")
ax.set_zlabel("Depth Layer", fontsize=12, fontweight="bold")
ax.set_title("3D End-to-End Viral Surveillance Pipeline", fontsize=16, fontweight="bold", pad=20)

# Hide axis ticks for cleaner look
ax.set_xticks(range(8))
ax.set_xticklabels([f"S{i+1}" for i in range(8)], fontsize=10)
ax.set_yticks([-1, 0, 1])
ax.set_yticklabels(["Track A", "Main", "Track B"], fontsize=9)
ax.set_zticks([-1, 0, 1])
ax.set_zticklabels(["Embedding", "Core", "Composition"], fontsize=9)

# View angle
ax.view_init(elev=22, azim=-55)

plt.tight_layout()
fig.savefig(OUT_DIR / "end_to_end_pipeline_3d.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "end_to_end_pipeline_3d.pdf", bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'end_to_end_pipeline_3d.png'}")
print(f"Saved: {OUT_DIR / 'end_to_end_pipeline_3d.pdf'}")
