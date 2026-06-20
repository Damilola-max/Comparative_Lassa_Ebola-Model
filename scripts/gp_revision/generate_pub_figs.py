#!/usr/bin/env python3
"""Generate publication-quality manuscript figures for GP revision."""
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from matplotlib.patches import Rectangle
from scipy.stats import entropy as scipy_entropy
from sklearn.decomposition import PCA

matplotlib.use("agg")

BASE = Path(__file__).resolve().parents[2]
RESULTS = BASE / "results" / "gp_revision"
ASSETS = BASE / "manuscript" / "assets" / "refined3_1" / "media"
ASSETS.mkdir(parents=True, exist_ok=True)

VIRUS_COLORS = {"Ebola": "#c62828", "Lassa": "#1565c0"}
CAT_COLORS = {
    "Critical": "#c62828",
    "Conserved": "#689f38",
    "Hotspot": "#fbc02d",
    "Intermediate": "#78909c",
    "MostlyGap": "#e0e0e0",
}


def ce(fasta, min_cov=0.5):
    """Compute conservation and entropy from aligned FASTA."""
    records = list(SeqIO.parse(fasta, "fasta"))
    seqs = [str(r.seq) for r in records]
    n_seq, length = len(seqs), len(seqs[0])
    cons, ents, valid = [], [], []
    for pos in range(length):
        col = [s[pos] for s in seqs]
        n_gap = col.count("-")
        if (n_seq - n_gap) / n_seq < min_cov:
            continue
        col_nogap = [c for c in col if c != "-"]
        n = len(col_nogap)
        counts = {}
        for c in col_nogap:
            counts[c] = counts.get(c, 0) + 1
        max_c = max(counts.values()) if counts else 0
        cons.append(max_c / n if n > 0 else 0)
        freqs = [c / n for c in counts.values()]
        ents.append(scipy_entropy(freqs, base=2) if freqs else 0)
        valid.append(pos)
    return np.array(valid), np.array(cons), np.array(ents)


def fig1():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    def box(x, y, w, h, t, c, fs=9):
        r = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            facecolor=c, edgecolor="#333", linewidth=1.2, alpha=0.92,
        )
        ax.add_patch(r)
        ax.text(
            x + w / 2, y + h / 2, t, ha="center", va="center",
            fontsize=fs, fontweight="bold",
            color="white" if c in ["#1565c0", "#c62828", "#37474f"] else "#333",
        )

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.3))

    ax.text(7, 8.6, "Classification & Risk Scoring Pipeline",
            ha="center", fontsize=15, fontweight="bold", color="#1565c0")

    for x, t in [(2.2, "STEP 1: Preprocessing"), (7, "STEP 2: Features"), (11.8, "STEP 3: Training")]:
        ax.text(x, 8.1, t, ha="center", fontsize=11, fontweight="bold", color="#1565c0")

    box(0.5, 6.8, 3.4, 0.9, "Raw Sequence\nFASTA / plain text", "#e3f2fd")
    arrow(2.2, 6.8, 2.2, 6.3)
    box(0.5, 5.3, 3.4, 0.8, "Clean Sequence\nRemove ambiguous chars", "#bbdefb")
    arrow(2.2, 5.3, 2.2, 4.8)
    box(0.5, 3.9, 3.4, 0.7, "ESM-2 Embedding\nesm2_t12_35M_UR50D", "#90caf9", 8)

    box(5.3, 6.8, 3.4, 0.9, "Feature Extraction\n21-dim composition", "#e8f5e9")
    arrow(7, 6.8, 7, 6.3)
    box(5.3, 5.3, 3.4, 0.8, "Concatenate\n480-dim ESM + 21 comp", "#c8e6c9")
    arrow(7, 5.3, 7, 4.8)
    box(5.3, 3.9, 3.4, 0.7, "Scale & Split\nStandardScaler + 80/20", "#a5d6a7", 8)

    box(10.3, 6.8, 3.0, 0.9, "Compare Models\nLR / SVM / RF / XGB", "#fce4ec")
    arrow(11.8, 6.8, 11.8, 6.3)
    box(10.3, 5.3, 3.0, 0.8, "Cross-Validation\n5-fold stratified", "#f8bbd9")
    arrow(11.8, 5.3, 11.8, 4.8)
    box(10.3, 3.9, 3.0, 0.7, "Best Model\nAcc = 1.000, AUC = 1.000", "#f48fb1")

    for y in [7.25, 5.65, 4.25]:
        arrow(3.9, y, 5.3, y)
        arrow(8.7, y, 10.3, y)

    box(1.0, 2.4, 3.5, 1.0, "Calibration\nClass centroids & distance stats\nAtypicality z-score", "#fff3e0")
    box(5.25, 2.4, 3.5, 1.0, "Inference\nLogReg predict_proba\nThreshold at 0.5", "#fff3e0")
    box(9.5, 2.4, 3.5, 1.0, "Risk Scoring\nOutlier detection\nIndex 0-100", "#fff3e0")
    arrow(4.5, 2.9, 5.25, 2.9)
    arrow(8.75, 2.9, 9.5, 2.9)

    outs = [
        ("Predicted\nVirus", "#c8e6c9"),
        ("Confidence\n(prob)", "#fff9c4"),
        ("Atypicality\nScore", "#ffe0b2"),
        ("Risk\nCategory", "#ffccbc"),
        ("Z-score\n(outlier)", "#ef9a9a"),
    ]
    for i, (t, c) in enumerate(outs):
        x = 0.7 + i * 2.6
        box(x, 0.6, 2.2, 1.0, t, c, 9)
        if i < 4:
            ax.annotate("", xy=(x + 2.5, 1.1), xytext=(x + 2.2, 1.1),
                        arrowprops=dict(arrowstyle="->", color="#999", lw=1))

    ax.text(7, 0.15, "Deployment: Streamlit App  |  https://mutation-analysis.streamlit.app",
            ha="center", fontsize=9, style="italic", color="#666")

    fig.savefig(ASSETS / "image1.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(ASSETS / "image1.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved image1.png")


def fig2():
    df = pd.read_csv(RESULTS / "site_categories.csv", header=None,
                     names=["virus", "pos", "score", "cat"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    cats = ["Critical", "Conserved", "Hotspot", "Intermediate"]

    ax = axes[0]
    ebo = df[df.virus == "Ebola"].cat.value_counts()
    las = df[df.virus == "Lassa"].cat.value_counts()
    ebo_tot = len(df[df.virus == "Ebola"])
    las_tot = len(df[df.virus == "Lassa"])
    x = np.arange(len(cats))
    w = 0.35
    ebo_v = [ebo.get(c, 0) / ebo_tot * 100 for c in cats]
    las_v = [las.get(c, 0) / las_tot * 100 for c in cats]
    b1 = ax.bar(x - w / 2, ebo_v, w, label="Ebola", color=VIRUS_COLORS["Ebola"],
                alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w / 2, las_v, w, label="Lassa", color=VIRUS_COLORS["Lassa"],
                alpha=0.85, edgecolor="white")
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if h > 1:
                ax.text(bar.get_x() + bar.get_width() / 2.0, h + 1,
                        f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Fraction of reference positions (%)", fontsize=12)
    ax.set_title("Site Category Distribution: Lassa vs Ebola", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=11)
    ax.legend(loc="upper right", frameon=True, fancybox=True)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2 = axes[1]
    ebo_s = [ebo.get(c, 0) / ebo_tot for c in cats]
    las_s = [las.get(c, 0) / las_tot for c in cats]
    colors = [CAT_COLORS[c] for c in cats]
    le, ll = 0, 0
    for i, (c, col) in enumerate(zip(cats, colors)):
        ax2.barh([1], [ebo_s[i]], left=[le], color=col, edgecolor="white", height=0.5)
        ax2.barh([0], [las_s[i]], left=[ll], color=col, edgecolor="white", height=0.5)
        if ebo_s[i] > 0.03:
            ax2.text(le + ebo_s[i] / 2, 1, f"{ebo_s[i] * 100:.0f}%",
                     ha="center", va="center", fontsize=9, fontweight="bold",
                     color="white" if col == "#c62828" else "#333")
        if las_s[i] > 0.03:
            ax2.text(ll + las_s[i] / 2, 0, f"{las_s[i] * 100:.0f}%",
                     ha="center", va="center", fontsize=9, fontweight="bold",
                     color="white" if col == "#c62828" else "#333")
        le += ebo_s[i]
        ll += las_s[i]
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Lassa", "Ebola"], fontsize=11)
    ax2.set_xlabel("Fraction of reference positions", fontsize=12)
    ax2.set_title("Stacked Site Category Composition", fontsize=13, fontweight="bold")
    ax2.set_xlim(0, 1.05)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    handles = [mpatches.Patch(color=CAT_COLORS[c], label=c) for c in cats]
    ax2.legend(handles=handles, loc="upper right", frameon=True, fancybox=True, fontsize=9)

    plt.tight_layout()
    fig.savefig(ASSETS / "image2.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(ASSETS / "image2.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved image2.png")


def fig3():
    df = pd.read_csv(RESULTS / "site_categories.csv", header=None,
                     names=["virus", "pos", "score", "cat"])
    fig, axes = plt.subplots(2, 1, figsize=(14, 4), sharex=True)
    for virus, idx in [("Lassa", 0), ("Ebola", 1)]:
        ax = axes[idx]
        sub = df[df.virus == virus].sort_values("pos")
        sub["pos"] = sub["pos"].astype(int)
        pos, cats = sub["pos"].values, sub["cat"].values
        for p, c in zip(pos, cats):
            ax.add_patch(Rectangle((p, 0), 1, 1, facecolor=CAT_COLORS.get(c, "#999"),
                                   edgecolor="none", alpha=0.92))
        ax.set_ylim(0, 1)
        ax.set_yticks([0.5])
        ax.set_yticklabels([virus], fontsize=12, fontweight="bold")
        ax.set_xlim(0, max(pos) + 10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(left=False, bottom=False)
        if idx == 1:
            ax.set_xlabel("Reference position", fontsize=12)
    handles = [mpatches.Patch(color=CAT_COLORS[c], label=c)
               for c in ["Critical", "Conserved", "Hotspot", "Intermediate"]]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               frameon=True, fancybox=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Site Category Barcode: Lassa vs Ebola", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(ASSETS / "image3.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(ASSETS / "image3.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved image3.png")


def fig4():
    ep, ec, ee = ce(RESULTS / "ebov_gp_sample.aligned.fasta")
    lp, lc, le = ce(RESULTS / "lasv_gp_sample.aligned.fasta")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.hist(lc, bins=30, alpha=0.7, color=VIRUS_COLORS["Lassa"], label="Lassa", edgecolor="white")
    ax.hist(ec, bins=30, alpha=0.7, color=VIRUS_COLORS["Ebola"], label="Ebola", edgecolor="white")
    ax.set_xlabel("Conservation (0-1)", fontsize=11)
    ax.set_ylabel("Number of sites", fontsize=11)
    ax.set_title("Conservation Distribution", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, fancybox=True)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[0, 1]
    ax.hist(le, bins=30, alpha=0.7, color=VIRUS_COLORS["Lassa"], label="Lassa", edgecolor="white")
    ax.hist(ee, bins=30, alpha=0.7, color=VIRUS_COLORS["Ebola"], label="Ebola", edgecolor="white")
    ax.set_xlabel("Shannon Entropy (bits)", fontsize=11)
    ax.set_ylabel("Number of sites", fontsize=11)
    ax.set_title("Entropy Distribution", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, fancybox=True)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for idx, (data, title, ylabel) in enumerate([
        ([lc, ec], "Conservation Violin", "Conservation"),
        ([le, ee], "Entropy Violin", "Entropy (bits)")
    ]):
        ax = axes[1, idx]
        parts = ax.violinplot(data, positions=[1, 2], showmeans=True, showmedians=True)
        for pc, col in zip(parts['bodies'], [VIRUS_COLORS["Lassa"], VIRUS_COLORS["Ebola"]]):
            pc.set_facecolor(col)
            pc.set_alpha(0.7)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Lassa", "Ebola"], fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Conservation & Entropy: Lassa vs Ebola", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(ASSETS / "image4.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(ASSETS / "image4.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved image4.png")


def fig5():
    ep, ec, ee = ce(RESULTS / "ebov_gp_sample.aligned.fasta")
    lp, lc, le = ce(RESULTS / "lasv_gp_sample.aligned.fasta")
    en = ep / max(ep) if len(ep) > 0 else ep
    ln = lp / max(lp) if len(lp) > 0 else lp
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax = axes[0]
    ax.plot(ln, lc, color=VIRUS_COLORS["Lassa"], alpha=0.7, lw=1, label="Lassa")
    ax.plot(en, ec, color=VIRUS_COLORS["Ebola"], alpha=0.7, lw=1, label="Ebola")
    ax.set_ylabel("Conservation (0-1)", fontsize=12)
    ax.set_title("Conservation Along Reference (Normalized Position)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", frameon=True, fancybox=True)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 1.05)

    ax = axes[1]
    ax.plot(ln, le, color=VIRUS_COLORS["Lassa"], alpha=0.7, lw=1, label="Lassa")
    ax.plot(en, ee, color=VIRUS_COLORS["Ebola"], alpha=0.7, lw=1, label="Ebola")
    ax.set_ylabel("Entropy (bits)", fontsize=12)
    ax.set_xlabel("Fractional Position (0 = N-terminus, 1 = C-terminus)", fontsize=12)
    ax.set_title("Entropy Along Reference (Normalized Position)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, fancybox=True)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(ASSETS / "image5.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(ASSETS / "image5.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved image5.png")


def fig6():
    emb = torch.load(RESULTS / "gp_embeddings.pt", weights_only=False)
    embeddings = emb["embeddings"].numpy()
    ids = emb["ids"]
    df_meta = pd.read_csv(BASE / "data" / "cleaned" / "cleaned_sequences_gp_only.csv")
    id_to_v = dict(zip(df_meta["id"].astype(str), df_meta["virus"]))
    labels = [id_to_v.get(str(i), "Unknown") for i in ids]
    pca = PCA(n_components=2)
    pca_emb = pca.fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(10, 7))

    for virus in ["Lassa", "Ebola"]:
        mask = np.array([l == virus for l in labels])
        ax.scatter(pca_emb[mask, 0], pca_emb[mask, 1], c=VIRUS_COLORS[virus],
                   label=virus, alpha=0.5, s=25, edgecolors="none")
    for virus in ["Lassa", "Ebola"]:
        mask = np.array([l == virus for l in labels])
        cx, cy = pca_emb[mask].mean(axis=0)
        ax.scatter(cx, cy, c="black", s=300, marker="*", zorder=10,
                   edgecolors="white", linewidths=2)
        ax.annotate(f"{virus} centroid", (cx, cy), fontsize=10, fontweight="bold",
                     xytext=(10, 10), textcoords="offset points",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)", fontsize=12)
    ax.set_title("PCA: ESM-2 Embedding Space — Lassa vs Ebola", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, fancybox=True)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(ASSETS / "image6.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(ASSETS / "image6.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved image6.png")


def fig7():
    emb = torch.load(RESULTS / "gp_embeddings.pt", weights_only=False)
    embeddings = emb["embeddings"].numpy()
    ids = emb["ids"]
    df_meta = pd.read_csv(BASE / "data" / "cleaned" / "cleaned_sequences_gp_only.csv")
    id_to_v = dict(zip(df_meta["id"].astype(str), df_meta["virus"]))
    labels = [id_to_v.get(str(i), "Unknown") for i in ids]
    ebo_mask = np.array([l == "Ebola" for l in labels])
    las_mask = np.array([l == "Lassa" for l in labels])
    ebo_c = embeddings[ebo_mask].mean(axis=0)
    las_c = embeddings[las_mask].mean(axis=0)
    ebo_d = np.linalg.norm(embeddings[ebo_mask] - ebo_c, axis=1)
    las_d = np.linalg.norm(embeddings[las_mask] - las_c, axis=1)
    ebo_s = (ebo_d - ebo_d.min()) / (ebo_d.max() - ebo_d.min()) * 100
    las_s = (las_d - las_d.min()) / (las_d.max() - las_d.min()) * 100
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.hist(las_s, bins=30, alpha=0.7, color=VIRUS_COLORS["Lassa"],
            label="Lassa", edgecolor="white")
    ax.hist(ebo_s, bins=30, alpha=0.7, color=VIRUS_COLORS["Ebola"],
            label="Ebola", edgecolor="white")
    ax.set_xlabel("ESM Outlier Score (0-100)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("ESM Outlier Score Distribution", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, fancybox=True)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    parts = ax.violinplot([las_s, ebo_s], positions=[1, 2], showmeans=True, showmedians=True)
    for pc, col in zip(parts['bodies'], [VIRUS_COLORS["Lassa"], VIRUS_COLORS["Ebola"]]):
        pc.set_facecolor(col)
        pc.set_alpha(0.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Lassa", "Ebola"], fontsize=11)
    ax.set_ylabel("Outlier Score (0-100)", fontsize=11)
    ax.set_xlabel("Virus", fontsize=11)
    ax.set_title("ESM Outlier Score: Violin Comparison", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("ESM-2 Sequence Outlier Analysis: Lassa vs Ebola",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(ASSETS / "image7.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(ASSETS / "image7.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved image7.png")


if __name__ == "__main__":
    print("=== Generating publication-quality figures ===")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    print(f"\n=== All 7 figures saved to {ASSETS} ===")
