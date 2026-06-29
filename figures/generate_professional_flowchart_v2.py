"""Professional pipeline flowchart."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon

OUT = Path(__file__).resolve().parents[0]
STEPS = OUT / "flow_steps_v2"
STEPS.mkdir(parents=True, exist_ok=True)

C = {"input": "#1565c0", "clean": "#00897b", "meta": "#6a1b9a", "feat_a": "#ef6c00",
     "feat_b": "#d84315", "model": "#c62828", "valid": "#455a64", "site": "#2e7d32",
     "dash": "#0277bd", "bg": "#f8f9fa", "text": "#263238", "gray": "#cfd8dc"}


def box(ax, x, y, w, h, color, title, lines, icon=None, fs=9, ts=10):
    ax.add_patch(FancyBboxPatch((x + 0.005, y - 0.005), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.02", facecolor="black", alpha=0.12, zorder=0, transform=ax.transAxes))
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.02", facecolor=color, edgecolor="white", linewidth=2, zorder=1, transform=ax.transAxes))
    ax.text(x + 0.015, y + h - 0.025, title, transform=ax.transAxes, fontsize=ts, fontweight="bold", color="white", va="top")
    for i, line in enumerate(lines):
        ax.text(x + 0.015, y + h - 0.065 - i * 0.035, f"• {line}", transform=ax.transAxes, fontsize=fs, color="white", va="top")
    if icon:
        iax = ax.inset_axes([x + w - 0.15, y + 0.015, 0.13, 0.13])
        icon(iax)
        iax.set_xlim(0, 1); iax.set_ylim(0, 1); iax.axis("off")


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#455a64", lw=2.2, connectionstyle="arc3,rad=0"), zorder=2)


def icon_globe(a):
    a.add_patch(Circle((0.5, 0.5), 0.35, fill=False, edgecolor="white", lw=2))
    a.plot([0.5, 0.5], [0.15, 0.85], color="white", lw=2)
    a.plot([0.15, 0.85], [0.5, 0.5], color="white", lw=2)


def icon_filter(a):
    a.add_patch(Polygon([[0.2, 0.8], [0.8, 0.8], [0.6, 0.45], [0.6, 0.1], [0.4, 0.1], [0.4, 0.45]], fill=False, edgecolor="white", lw=2.5))


def icon_doc(a):
    a.add_patch(Rectangle((0.15, 0.1), 0.5, 0.7, fill=False, edgecolor="white", lw=2))
    a.plot([0.22, 0.58], [0.55, 0.55], color="white", lw=1.5)
    a.plot([0.22, 0.58], [0.42, 0.42], color="white", lw=1.5)
    a.add_patch(Circle((0.72, 0.72), 0.12, fill=False, edgecolor="white", lw=2))


def icon_grid(a):
    for i in range(5):
        for j in range(5):
            a.add_patch(Rectangle((0.1 + i * 0.16, 0.1 + j * 0.16), 0.13, 0.13,
                         facecolor="white", alpha=0.2 + 0.6 * np.random.random(), edgecolor="white"))


def icon_net(a):
    layers = [3, 5, 4, 2]
    xs = [0.15, 0.40, 0.65, 0.88]
    nodes = []
    for n, x in zip(layers, xs):
        ys = np.linspace(0.15, 0.85, n)
        layer = []
        for y in ys:
            a.add_patch(Circle((x, y), 0.055, facecolor="white", edgecolor="white"))
            layer.append((x, y))
        nodes.append(layer)
    for i in range(len(nodes) - 1):
        for p in nodes[i]:
            for q in nodes[i + 1]:
                a.plot([p[0], q[0]], [p[1], q[1]], color="white", lw=0.8, alpha=0.5)


def icon_tree(a):
    a.plot([0.5, 0.5], [0.82, 0.55], color="white", lw=2.5)
    a.plot([0.5, 0.25], [0.55, 0.35], color="white", lw=2.5)
    a.plot([0.5, 0.75], [0.55, 0.35], color="white", lw=2.5)
    for x in [0.15, 0.35, 0.65, 0.85]:
        a.add_patch(Rectangle((x - 0.05, 0.08), 0.10, 0.08, facecolor="white"))
    a.add_patch(Circle((0.5, 0.82), 0.05, facecolor="white"))


def icon_flask(a):
    a.plot([0.35, 0.35], [0.75, 0.45], color="white", lw=2.5)
    a.plot([0.35, 0.2], [0.45, 0.15], color="white", lw=2.5)
    a.plot([0.35, 0.65], [0.45, 0.15], color="white", lw=2.5)
    a.plot([0.65, 0.65], [0.45, 0.75], color="white", lw=2.5)
    a.plot([0.3, 0.7], [0.75, 0.75], color="white", lw=2.5)
    a.add_patch(Circle((0.72, 0.72), 0.06, facecolor="white"))


def icon_dna(a):
    t = np.linspace(0, 2 * np.pi, 100)
    a.plot(0.5 + 0.25 * np.sin(t), 0.5 + 0.35 * np.cos(t), color="white", lw=2.5)
    a.plot(0.5 - 0.25 * np.sin(t), 0.5 + 0.35 * np.cos(t), color="white", lw=2.5)


def icon_dash(a):
    a.add_patch(Rectangle((0.1, 0.1), 0.8, 0.75, fill=False, edgecolor="white", lw=2.5))
    a.plot([0.1, 0.9], [0.78, 0.78], color="white", lw=2.5)
    a.add_patch(Rectangle((0.18, 0.22), 0.18, 0.38, facecolor="white"))
    a.add_patch(Rectangle((0.44, 0.32), 0.18, 0.28, facecolor="white"))
    a.plot([0.18, 0.36, 0.62, 0.80], [0.35, 0.5, 0.42, 0.6], color="white", lw=2)


ICONS = {
    "globe": icon_globe, "filter": icon_filter, "doc": icon_doc, "grid": icon_grid,
    "net": icon_net, "tree": icon_tree, "flask": icon_flask, "dna": icon_dna, "dash": icon_dash,
}


def save_step(filename, title, color, icon_key, lines):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor(C["bg"])
    iax = ax.inset_axes([0.25, 0.45, 0.5, 0.45])
    ICONS[icon_key](iax)
    iax.set_xlim(0, 1); iax.set_ylim(0, 1); iax.axis("off")
    ax.text(0.5, 0.30, title, fontsize=18, fontweight="bold", color=color, ha="center", va="top")
    for i, line in enumerate(lines):
        ax.text(0.5, 0.22 - i * 0.06, f"• {line}", fontsize=11, color=C["text"], ha="center", va="top")
    fig.savefig(STEPS / filename, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)


def build_combined():
    fig, ax = plt.subplots(figsize=(30, 18), dpi=150)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor(C["bg"])

    ax.add_patch(Rectangle((0.02, 0.92), 0.96, 0.05, facecolor=C["gray"], edgecolor="none"))
    ax.text(0.5, 0.945, "End-to-End Viral Surveillance & Classification Pipeline",
            fontsize=26, fontweight="bold", color=C["text"], ha="center", va="center")

    bw, bh = 0.19, 0.24
    gap = 0.025
    r1 = 0.60
    r2 = 0.18
    x = [0.03 + i * (bw + gap) for i in range(5)]
    hw = (bw - gap) / 2

    # Row 1
    box(ax, x[0], r1, bw, bh, C["input"], "1. Data Acquisition", ["NCBI / EpiFlu", "Raw FASTA files"], ICONS["globe"])
    box(ax, x[1], r1, bw, bh, C["clean"], "2. Sequence Cleaning", ["Remove X/B/Z", "Strip U/O", "Drop < 50 AA"], ICONS["filter"])
    box(ax, x[2], r1, bw, bh, C["meta"], "3. Metadata Parsing", ["Accession", "Country / Date", "Lineage"], ICONS["doc"])
    box(ax, x[3], r1, hw, bh, C["feat_a"], "4A. Composition", ["20 AA frequencies", "Sequence length"], ICONS["grid"], fs=8, ts=9)
    box(ax, x[3] + hw + gap, r1, hw, bh, C["feat_b"], "4B. ESM-2", ["35M model", "480-dim embed"], ICONS["net"], fs=8, ts=9)

    # Row 2
    box(ax, x[1], r2, bw, bh, C["model"], "5. Model & Calibration", ["Random Forest / Logistic", "Class centroids", "Probability calibration"], ICONS["tree"])
    box(ax, x[2], r2, bw, bh, C["valid"], "6. Validation & QA", ["Stratified CV", "Grouped CV", "Ablation / edge cases"], ICONS["flask"])
    box(ax, x[3], r2, bw, bh, C["site"], "7. Site-Level Analysis", ["Shannon entropy", "GP alignment map", "Mutational profiling"], ICONS["dna"])
    box(ax, x[4], r2, bw, bh, C["dash"], "8. Streamlit Dashboard", ["Drag-and-drop FASTA", "Real-time risk charts", "CSV / PDF export"], ICONS["dash"])

    # Arrows row 1
    arrow(ax, x[0] + bw, r1 + bh / 2, x[1], r1 + bh / 2)
    arrow(ax, x[1] + bw, r1 + bh / 2, x[2], r1 + bh / 2)
    arrow(ax, x[2] + bw, r1 + bh / 2, x[3], r1 + bh / 2)
    arrow(ax, x[3] + bw, r1 + bh / 2, x[3] + hw + gap, r1 + bh / 2)

    # Down to row 2
    arrow(ax, x[3] + hw + gap / 2, r1, x[3] + hw + gap / 2, r2 + bh)

    # Arrows row 2
    arrow(ax, x[1] + bw, r2 + bh / 2, x[2], r2 + bh / 2)
    arrow(ax, x[2] + bw, r2 + bh / 2, x[3], r2 + bh / 2)
    arrow(ax, x[3] + bw, r2 + bh / 2, x[4], r2 + bh / 2)

    ax.text(0.5, 0.02, "Reproducible: end_to_end_pipeline.py  |  Deployed: mutation-analysis.streamlit.app",
            fontsize=11, color=C["text"], ha="center", style="italic")

    fig.savefig(OUT / "end_to_end_pipeline_professional_v2.png", dpi=200, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)


if __name__ == "__main__":
    save_step("step1.png", "1. Data Acquisition", C["input"], "globe", ["NCBI / EpiFlu / Manuscripts", "Raw FASTA consolidation"])
    save_step("step2.png", "2. Sequence Cleaning", C["clean"], "filter", ["Remove X/B/Z wildcards", "Strip U/O residues", "Drop fragments < 50 AA"])
    save_step("step3.png", "3. Metadata Parsing", C["meta"], "doc", ["Extract accession", "Country / collection date", "Lineage / organism"])
    save_step("step4a.png", "4A. Composition", C["feat_a"], "grid", ["20 canonical AA frequencies", "Sequence length"])
    save_step("step4b.png", "4B. ESM-2", C["feat_b"], "net", ["facebook/esm2_t12_35M", "480-dim mean-pooled embeddings"])
    save_step("step5.png", "5. Model & Calibration", C["model"], "tree", ["Random Forest / Logistic", "Class centroids", "Isotonic calibration"])
    save_step("step6.png", "6. Validation & QA", C["valid"], "flask", ["Stratified cross-validation", "Grouped cross-validation", "Ablation & edge-case panels"])
    save_step("step7.png", "7. Site-Level Analysis", C["site"], "dna", ["Shannon entropy profile", "GP alignment mapping", "Mutational profiling"])
    save_step("step8.png", "8. Streamlit Dashboard", C["dash"], "dash", ["Drag-and-drop FASTA", "Real-time risk charts", "CSV & PDF export"])
    build_combined()
    print(f"Saved: {OUT / 'end_to_end_pipeline_professional_v2.png'}")
