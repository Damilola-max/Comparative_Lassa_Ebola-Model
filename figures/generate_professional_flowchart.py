"""Professional publication-quality pipeline flowchart (PNG)."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon

OUT = Path(__file__).resolve().parents[0]
OUT.mkdir(parents=True, exist_ok=True)
STEPS = OUT / "flow_steps"
STEPS.mkdir(parents=True, exist_ok=True)

C = {
    "input": "#1565c0", "clean": "#00897b", "meta": "#6a1b9a",
    "feat_a": "#ef6c00", "feat_b": "#d84315", "model": "#c62828",
    "valid": "#455a64", "site": "#2e7d32", "dash": "#0277bd",
    "bg": "#f8f9fa", "text": "#263238", "gray": "#cfd8dc",
}


def box(ax, x, y, w, h, color, title, lines, icon=None, fs=9, ts=11):
    shadow = FancyBboxPatch((x + 0.005, y - 0.005), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor="black", alpha=0.12, zorder=0, transform=ax.transAxes)
    ax.add_patch(shadow)
    rect = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor=color, edgecolor="white", linewidth=2, zorder=1, transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(x + 0.015, y + h - 0.02, title, transform=ax.transAxes,
            fontsize=ts, fontweight="bold", color="white", va="top")
    for i, line in enumerate(lines):
        ax.text(x + 0.015, y + h - 0.06 - i * 0.038, f"• {line}",
                transform=ax.transAxes, fontsize=fs, color="white", va="top")
    if icon:
        iax = ax.inset_axes([x + w - 0.16, y + 0.015, 0.14, 0.14])
        icon(iax)
        iax.set_xlim(0, 1)
        iax.set_ylim(0, 1)
        iax.axis("off")


def arrow(ax, x1, y1, x2, y2, c="#455a64"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=c, lw=2.2,
                               connectionstyle="arc3,rad=0"), zorder=2)


def icon_globe(ax):
    ax.add_patch(Circle((0.5, 0.5), 0.35, fill=False, edgecolor="white", lw=2))
    ax.plot([0.5, 0.5], [0.15, 0.85], color="white", lw=2)
    ax.plot([0.15, 0.85], [0.5, 0.5], color="white", lw=2)
    ax.plot([0.22, 0.78], [0.32, 0.68], color="white", lw=1.5)
    ax.plot([0.22, 0.78], [0.68, 0.32], color="white", lw=1.5)


def icon_filter(ax):
    ax.add_patch(Polygon([[0.2, 0.8], [0.8, 0.8], [0.6, 0.45], [0.6, 0.1], [0.4, 0.1], [0.4, 0.45]],
                   fill=False, edgecolor="white", lw=2.5))
    ax.plot([0.3, 0.25], [0.5, 0.4], color="white", lw=2)
    ax.plot([0.4, 0.35], [0.5, 0.4], color="white", lw=2)
    ax.plot([0.5, 0.45], [0.5, 0.4], color="white", lw=2)


def icon_doc(ax):
    ax.add_patch(Rectangle((0.15, 0.1), 0.5, 0.7, fill=False, edgecolor="white", lw=2))
    ax.plot([0.22, 0.58], [0.55, 0.55], color="white", lw=1.5)
    ax.plot([0.22, 0.58], [0.42, 0.42], color="white", lw=1.5)
    ax.plot([0.22, 0.45], [0.29, 0.29], color="white", lw=1.5)
    ax.add_patch(Circle((0.72, 0.72), 0.12, fill=False, edgecolor="white", lw=2))
    ax.plot([0.81, 0.9], [0.63, 0.52], color="white", lw=2.5)


def icon_grid(ax):
    for i in range(5):
        for j in range(5):
            alpha = 0.2 + 0.6 * np.random.random()
            ax.add_patch(Rectangle((0.1 + i * 0.16, 0.1 + j * 0.16), 0.13, 0.13,
                          facecolor="white", alpha=alpha, edgecolor="white"))


def icon_net(ax):
    layers = [3, 5, 4, 2]
    xs = [0.15, 0.40, 0.65, 0.88]
    nodes = []
    for layer_idx, n in enumerate(layers):
        ys = np.linspace(0.15, 0.85, n)
        layer = []
        for y in ys:
            ax.add_patch(Circle((xs[layer_idx], y), 0.055, facecolor="white", edgecolor="white"))
            layer.append((xs[layer_idx], y))
        nodes.append(layer)
    for i in range(len(nodes) - 1):
        for a in nodes[i]:
            for b in nodes[i + 1]:
                ax.plot([a[0], b[0]], [a[1], b[1]], color="white", lw=0.8, alpha=0.5)


def icon_tree(ax):
    ax.plot([0.5, 0.5], [0.82, 0.55], color="white", lw=2.5)
    ax.plot([0.5, 0.25], [0.55, 0.35], color="white", lw=2.5)
    ax.plot([0.5, 0.75], [0.55, 0.35], color="white", lw=2.5)
    for x in [0.25, 0.75]:
        ax.plot([x, x - 0.1], [0.35, 0.15], color="white", lw=2.5)
        ax.plot([x, x + 0.1], [0.35, 0.15], color="white", lw=2.5)
    for x in [0.15, 0.35, 0.65, 0.85]:
        ax.add_patch(Rectangle((x - 0.05, 0.08), 0.10, 0.08, facecolor="white"))
    ax.add_patch(Circle((0.5, 0.82), 0.05, facecolor="white"))


def icon_flask(ax):
    ax.plot([0.35, 0.35], [0.75, 0.45], color="white", lw=2.5)
    ax.plot([0.35, 0.2], [0.45, 0.15], color="white", lw=2.5)
    ax.plot([0.35, 0.65], [0.45, 0.15], color="white", lw=2.5)
    ax.plot([0.65, 0.65], [0.45, 0.75], color="white", lw=2.5)
    ax.plot([0.3, 0.7], [0.75, 0.75], color="white", lw=2.5)
    ax.plot([0.22, 0.78], [0.6, 0.6], color="white", lw=1.5, linestyle="--")
    ax.add_patch(Circle((0.72, 0.72), 0.06, facecolor="white"))
    ax.plot([0.77, 0.88], [0.67, 0.56], color="white", lw=2.5)


def icon_dna(ax):
    t = np.linspace(0, 2 * np.pi, 100)
    ax.plot(0.5 + 0.25 * np.sin(t), 0.5 + 0.35 * np.cos(t), color="white", lw=2.5)
    ax.plot(0.5 - 0.25 * np.sin(t), 0.5 + 0.35 * np.cos(t), color="white", lw=2.5)
    for p in np.linspace(0, 2 * np.pi, 7):
        ax.plot([0.5 + 0.25 * np.sin(p), 0.5 - 0.25 * np.sin(p)],
                [0.5 + 0.35 * np.cos(p), 0.5 + 0.35 * np.cos(p)], color="white", lw=1.5)


def icon_dash(ax):
    ax.add_patch(Rectangle((0.1, 0.1), 0.8, 0.75, fill=False, edgecolor="white", lw=2.5))
    ax.plot([0.1, 0.9], [0.78, 0.78], color="white", lw=2.5)
    ax.add_patch(Circle((0.78, 0.86), 0.03, facecolor="white"))
    ax.add_patch(Circle((0.86, 0.86), 0.03, facecolor="white"))
    ax.add_patch(Rectangle((0.18, 0.22), 0.18, 0.38, facecolor="white"))
    ax.add_patch(Rectangle((0.44, 0.32), 0.18, 0.28, facecolor="white"))
    ax.plot([0.18, 0.36, 0.62, 0.80], [0.35, 0.5, 0.42, 0.6], color="white", lw=2)


def save_step(filename, title, color, icon, lines):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])
    iax = ax.inset_axes([0.25, 0.45, 0.5, 0.45])
    icon(iax)
    iax.set_xlim(0, 1)
    iax.set_ylim(0, 1)
    iax.axis("off")
    ax.text(0.5, 0.30, title, fontsize=18, fontweight="bold", color=color, ha="center", va="top")
    for i, line in enumerate(lines):
        ax.text(0.5, 0.22 - i * 0.06, f"• {line}", fontsize=11, color=C["text"], ha="center", va="top")
    fig.savefig(STEPS / filename, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)


def build_combined():
    fig, ax = plt.subplots(figsize=(22, 14), dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])

    # Header
    ax.add_patch(Rectangle((0.02, 0.90), 0.96, 0.07, facecolor=C["gray"], edgecolor="none"))
    ax.text(0.5, 0.935, "End-to-End Viral Surveillance & Classification Pipeline",
            fontsize=22, fontweight="bold", color=C["text"], ha="center", va="center")

    # Section labels
    ax.text(0.14, 0.83, "Data Ingest", fontsize=14, fontweight="bold", color=C["text"], ha="center")
    ax.text(0.38, 0.83, "Feature Engineering", fontsize=14, fontweight="bold", color=C["text"], ha="center")
    ax.text(0.62, 0.83, "Modeling & Calibration", fontsize=14, fontweight="bold", color=C["text"], ha="center")
    ax.text(0.86, 0.83, "Validation & Deployment", fontsize=14, fontweight="bold", color=C["text"], ha="center")

    bw, bh = 0.18, 0.22

    # Column 1
    box(ax, 0.03, 0.52, bw, bh, C["input"], "1. Data Acquisition", ["NCBI / EpiFlu", "Raw FASTA files"], icon_globe)
    box(ax, 0.03, 0.27, bw, bh, C["clean"], "2. Sequence Cleaning", ["Remove X/B/Z", "Strip U/O", "Drop < 50 AA"], icon_filter)
    box(ax, 0.03, 0.02, bw, bh, C["meta"], "3. Metadata Parsing", ["Accession", "Country / Date", "Lineage"], icon_doc)

    # Column 2: split feature engineering
    hw = (bw - 0.01) / 2
    box(ax, 0.27, 0.40, hw, 0.30, C["feat_a"], "4A. Composition", ["20 AA frequencies", "Sequence length"], icon_grid, fs=8, ts=9)
    box(ax, 0.27 + hw + 0.01, 0.40, hw, 0.30, C["feat_b"], "4B. ESM-2", ["35M parameter model", "480-dim embedding"], icon_net, fs=8, ts=9)

    # Column 3
    box(ax, 0.51, 0.40, bw, 0.30, C["model"], "5. Model & Calibration", ["Random Forest / Logistic", "Class centroids", "Probability calibration"], icon_tree, fs=9, ts=10)

    # Column 4: stacked validation, site, dashboard
    box(ax, 0.75, 0.52, bw, bh, C["valid"], "6. Validation & QA", ["Stratified CV", "Grouped CV", "Ablation / edge cases"], icon_flask)
    box(ax, 0.75, 0.27, bw, bh, C["site"], "7. Site-Level Analysis", ["Shannon entropy", "GP alignment map", "Mutational profiling"], icon_dna)
    box(ax, 0.75, 0.02, bw, bh, C["dash"], "8. Streamlit Dashboard", ["Drag-and-drop FASTA", "Real-time risk charts", "CSV / PDF export"], icon_dash)

    # Arrows between columns
    # Ingest -> Feature Engineering
    arrow(ax, 0.21, 0.63, 0.27, 0.55)
    arrow(ax, 0.21, 0.38, 0.27, 0.55)
    arrow(ax, 0.21, 0.13, 0.27, 0.55)

    # Feature Engineering -> Model
    arrow(ax, 0.45, 0.55, 0.51, 0.55)
    arrow(ax, 0.45, 0.55, 0.51, 0.55)

    # Model -> Validation/Deployment
    arrow(ax, 0.69, 0.55, 0.75, 0.63)
    arrow(ax, 0.69, 0.55, 0.75, 0.38)
    arrow(ax, 0.69, 0.55, 0.75, 0.13)

    # Vertical arrows in column 4
    arrow(ax, 0.84, 0.52, 0.84, 0.49)
    arrow(ax, 0.84, 0.27, 0.84, 0.24)

    # Annotations
    ax.text(0.50, 0.02, "Pipeline reproducible from: end_to_end_pipeline.py  |  Deployed at: mutation-analysis.streamlit.app",
            fontsize=10, color=C["text"], ha="center", style="italic")

    fig.savefig(OUT / "end_to_end_pipeline_professional.png", dpi=200, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)


if __name__ == "__main__":
    save_step("step1_data_acquisition.png", "1. Data Acquisition", C["input"], icon_globe,
              ["NCBI / EpiFlu / Manuscripts", "Raw FASTA consolidation"])
    save_step("step2_sequence_cleaning.png", "2. Sequence Cleaning", C["clean"], icon_filter,
              ["Remove X/B/Z wildcards", "Strip U/O residues", "Drop fragments < 50 AA"])
    save_step("step3_metadata_parsing.png", "3. Metadata Parsing", C["meta"], icon_doc,
              ["Extract accession", "Country / collection date", "Lineage / organism"])
    save_step("step4a_composition.png", "4A. Composition Features", C["feat_a"], icon_grid,
              ["20 canonical AA frequencies", "Sequence length"])
    save_step("step4b_esm2.png", "4B. ESM-2 Embeddings", C["feat_b"], icon_net,
              ["facebook/esm2_t12_35M", "480-dim mean-pooled embeddings"])
    save_step("step5_model_calibration.png", "5. Model & Calibration", C["model"], icon_tree,
              ["Random Forest / Logistic", "Class centroids", "Isotonic calibration"])
    save_step("step6_validation_qa.png", "6. Validation & QA", C["valid"], icon_flask,
              ["Stratified cross-validation", "Grouped cross-validation", "Ablation & edge-case panels"])
    save_step("step7_site_analysis.png", "7. Site-Level Analysis", C["site"], icon_dna,
              ["Shannon entropy profile", "GP alignment mapping", "Mutational profiling"])
    save_step("step8_dashboard.png", "8. Streamlit Dashboard", C["dash"], icon_dash,
              ["Drag-and-drop FASTA", "Real-time risk charts", "CSV & PDF export"])

    build_combined()
    print(f"Saved individual steps to {STEPS}")
    print(f"Saved combined flowchart to {OUT / 'end_to_end_pipeline_professional.png'}")
