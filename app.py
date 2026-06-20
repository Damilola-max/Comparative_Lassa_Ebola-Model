from io import StringIO, BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from Bio import SeqIO
    BIOPYTHON_AVAILABLE = True
except Exception:
    BIOPYTHON_AVAILABLE = False

# Matplotlib for rich charts
try:
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False
    plt = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

from src.config import METRICS_PATH, MODEL_PATH
from src.features.sequence_features import clean_sequence
from src.models.predict import predict_sequences

st.set_page_config(page_title="Lassa vs Ebola GP Sequence Classifier", layout="wide")

# ── Header ──
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("Comparative Lassa–Ebola GP Sequence Classifier")
    st.markdown(
        "**ESM-2 + Composition Ensemble Model** — "
        "Upload glycoprotein sequences to get virus classification, confidence, atypicality scoring, and composition deviation."
    )
with header_col2:
    st.markdown(
        """
        <div style="text-align:right; font-size:12px; color:#666;">
            <b>Model:</b> Composition features (length + AA freq)<br>
            <b>Accuracy:</b> 1.000<br>
            <b>Outlier Detection:</b> Enabled
        </div>
        """,
        unsafe_allow_html=True,
    )

st.info(
    "**How it works:** The classifier uses lightweight composition features (sequence length + amino-acid "
    "frequencies) for virus identification. Atypicality measures how far a sequence sits from known training "
    "patterns. The model flags highly atypical sequences (≥95 index or z≥3.0) as **Unknown**. "
    "ESM-2 embeddings are used for descriptive embedding-space analysis only, not for classification."
)

st.warning(
    "**Intended use:** This application is a research and educational prototype for comparative sequence "
    "analysis. It is not a validated clinical or surveillance tool. Predictions should not be used for "
    "patient diagnosis, treatment selection, or public-health decisions without independent experimental validation."
)


# ──────────────────────────────────────────────
# Helper styles
# ──────────────────────────────────────────────
def _confidence_band(confidence: float) -> str:
    if confidence >= 0.95:
        return "very high"
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.70:
        return "moderate"
    return "low"


def _atypicality_phrase(z: float) -> str:
    if z <= -1.0:
        return "very typical of known examples"
    if z <= 0.5:
        return "within the usual range of known examples"
    if z <= 1.5:
        return "slightly unusual compared with known examples"
    return "strongly unusual compared with known examples"


def _explain_prediction(row: dict) -> str:
    confidence_pct = row["confidence"] * 100
    pred = row["predicted_virus"]
    if "Unknown" in pred:
        return (
            f"Sequence {row['id']} is **HIGHLY ATYPICAL** and does not match known "
            f"Ebola or Lassa patterns (confidence {confidence_pct:.1f}%). "
            f"Its atypicality index is {row['atypicality_index']:.1f}/100, which maps to the "
            f"'{row['atypicality_band']}' band. This sequence may be from a different virus, "
            f"contain significant mutations, or be a synthetic/artifact sequence. "
            f"Mutation risk score: {row.get('mutation_risk_score', 0):.1f}/100."
        )
    return (
        f"Sequence {row['id']} was classified as **{pred}** with {confidence_pct:.1f}% confidence "
        f"({_confidence_band(row['confidence'])} confidence). "
        f"Its atypicality index is {row['atypicality_index']:.1f}/100, which maps to the "
        f"'{row['atypicality_band']}' band. "
        f"The atypicality z-score is {row['atypicality_zscore']:.2f}, meaning this sequence is "
        f"{_atypicality_phrase(row['atypicality_zscore'])}. "
        f"Mutation risk score: {row.get('mutation_risk_score', 0):.1f}/100."
    )


# ──────────────────────────────────────────────
# Color maps
# ──────────────────────────────────────────────
ATYP_COLORS = {
    "Low": "#2e7d32",
    "Below-Average": "#689f38",
    "Average": "#fbc02d",
    "Elevated": "#f57c00",
    "High": "#c62828",
}

VIRUS_COLORS = {"Lassa": "#1565c0", "Ebola": "#c62828", "Unknown / Highly Atypical": "#455a64"}


def _band_color(band: str):
    return ATYP_COLORS.get(band, "#455a64")


# ──────────────────────────────────────────────
# Matplotlib charts
# ──────────────────────────────────────────────
def _fig_atypicality_gauge(value: float, band: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 2.5))
    # Background bar (0-100)
    ax.barh([0], [100], color="#e0e0e0", height=0.4, left=0)
    # Color zones
    zones = [(0, 20, "#2e7d32"), (20, 40, "#689f38"), (40, 60, "#fbc02d"),
             (60, 80, "#f57c00"), (80, 100, "#c62828")]
    for z0, z1, zc in zones:
        ax.barh([0], [z1 - z0], color=zc, height=0.4, left=z0, alpha=0.6)
    # Value marker
    ax.axvline(value, color="black", linewidth=3)
    ax.scatter([value], [0], color="black", s=150, zorder=5)
    ax.text(value, 0.28, f"{value:.1f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(["0", "20", "40", "60", "80", "100"])
    ax.set_xlabel("Atypicality Index (0-100)", fontsize=11)
    ax.set_title(f"Atypicality: {band}", fontsize=13, fontweight="bold", color=_band_color(band))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.tight_layout()
    return fig


def _fig_class_distribution(result_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = result_df["predicted_virus"].value_counts()
    colors = [VIRUS_COLORS.get(v, "#455a64") for v in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=2)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Predicted Class Distribution", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def _fig_band_distribution(result_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    band_order = ["Low", "Below-Average", "Average", "Elevated", "High"]
    counts = result_df["atypicality_band"].value_counts().reindex(band_order, fill_value=0)
    colors = [_band_color(b) for b in counts.index]
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="white", linewidth=2)
    ax.set_xlabel("Count", fontsize=11)
    ax.set_title("Atypicality Band Distribution", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, counts.values):
        if val > 0:
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    str(int(val)), ha="left", va="center", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def _fig_confidence_vs_atypicality(result_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    for virus in result_df["predicted_virus"].unique():
        subset = result_df[result_df["predicted_virus"] == virus]
        ax.scatter(subset["atypicality_index"], subset["confidence"],
                   c=VIRUS_COLORS.get(virus, "#455a64"), label=virus,
                   s=120, alpha=0.8, edgecolors="white", linewidth=1.5)
    ax.set_xlabel("Atypicality Index", fontsize=11)
    ax.set_ylabel("Confidence", fontsize=11)
    ax.set_title("Confidence vs Atypicality", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Predicted", loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def _fig_atypicality_per_sequence(result_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [_band_color(b) for b in result_df["atypicality_band"]]
    ax.bar(range(len(result_df)), result_df["atypicality_index"], color=colors,
           edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(result_df)))
    ax.set_xticklabels(result_df["id"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Atypicality Index", fontsize=11)
    ax.set_title("Atypicality Index per Sequence", fontsize=12, fontweight="bold")
    ax.axhline(50, color="black", linestyle="--", alpha=0.3, label="Baseline (50)")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def _fig_composition_radar(row: dict) -> plt.Figure:
    # Simple amino-acid composition bar chart for the sequence
    seq = row.get("sequence", "")
    if not seq:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Composition data not available", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig
    aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    counts = {aa: seq.count(aa) for aa in aa_list}
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(aa_list, [counts[aa] for aa in aa_list], color="#455a64", edgecolor="white")
    ax.set_xlabel("Amino Acid", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"Amino Acid Composition — {row['id'][:40]}", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────
# Report card (enhanced)
# ──────────────────────────────────────────────
def _render_report_card(row: dict):
    color = _band_color(row["atypicality_band"])
    confidence_pct = row["confidence"] * 100
    is_unknown = "Unknown" in row["predicted_virus"]
    pred_color = VIRUS_COLORS.get(row["predicted_virus"], "#455a64")

    st.markdown("---")
    st.subheader("Detailed Report Card")

    # Prediction badge (prominent)
    st.markdown(
        f"""
        <div style="background:{pred_color}15; border:2px solid {pred_color}; border-radius:12px; padding:14px 20px; margin-bottom:16px;">
            <div style="font-size:28px; font-weight:bold; color:{pred_color};">
                {row['predicted_virus']}
            </div>
            <div style="font-size:13px; color:#555;">
                Confidence: <b>{confidence_pct:.1f}%</b> &nbsp;|&nbsp;
                Length: <b>{row['sequence_length']}</b> aa &nbsp;|&nbsp;
                Composition Deviation: <b>{row.get('mutation_risk_score', 0):.1f}/100</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Warning banner for Unknown
    if is_unknown:
        st.error(
            "**HIGHLY ATYPICAL SEQUENCE DETECTED** — This sequence does not match known Ebola or Lassa "
            "glycoprotein patterns. It may be from a different virus, contain major mutations, or be synthetic."
        )

    # Metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Confidence", f"{confidence_pct:.1f}%", _confidence_band(row["confidence"]))
    c2.metric("Atypicality", f"{row['atypicality_index']:.1f}/100", row["atypicality_band"])
    c3.metric("Z-Score", f"{row['atypicality_zscore']:.2f}", _atypicality_phrase(row["atypicality_zscore"]))
    c4.metric("Comp Deviation", f"{row.get('mutation_risk_score', 0):.1f}/100")
    c5.metric("Seq Length", f"{row['sequence_length']} aa")

    # Gauge
    if MATPLOTLIB_AVAILABLE:
        st.pyplot(_fig_atypicality_gauge(row["atypicality_index"], row["atypicality_band"]), use_container_width=True)

    # Explanation box
    st.markdown(
        f"""
        <div style="border-left: 5px solid {color}; padding: 12px 16px; background-color: #fafafa; border-radius: 0 8px 8px 0;">
            <b>Interpretation:</b> {_explain_prediction(row)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Composition chart
    if "sequence" in row:
        st.pyplot(_fig_composition_radar(row), use_container_width=True)


# ──────────────────────────────────────────────
# PDF Report (enhanced with ReportLab tables)
# ──────────────────────────────────────────────
def _build_pdf_report(result_df: pd.DataFrame) -> bytes:
    if not REPORTLAB_AVAILABLE:
        return b""

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("<b>Comparative Lassa-Ebola Sequence Prediction Report</b>", styles["Title"]))
    story.append(Paragraph(f"Total sequences analyzed: <b>{len(result_df)}</b>", styles["Normal"]))
    story.append(Spacer(1, 0.5*cm))

    # Summary table
    summary_data = [["Metric", "Value"]]
    class_counts = result_df["predicted_virus"].value_counts().to_dict()
    for v, c in class_counts.items():
        summary_data.append([f"Predicted {v}", str(c)])
    band_counts = result_df["atypicality_band"].value_counts().to_dict()
    for b, c in band_counts.items():
        summary_data.append([f"Band: {b}", str(c)])
    summary_data.append(["Mean Atypicality Index", f"{result_df['atypicality_index'].mean():.2f}"])
    summary_data.append(["Mean Confidence", f"{result_df['confidence'].mean():.3f}"])

    summary_table = Table(summary_data, colWidths=[8*cm, 6*cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565c0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f5f5f5")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.8*cm))

    # Per-sequence detail table
    story.append(Paragraph("<b>Per-Sequence Results</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.3*cm))

    detail_data = [["ID", "Predicted", "Confidence", "Atypicality", "Band", "Z-Score"]]
    for row in result_df.to_dict(orient="records"):
        detail_data.append([
            str(row["id"])[:30],
            row["predicted_virus"],
            f"{row['confidence']:.3f}",
            f"{row['atypicality_index']:.1f}",
            row["atypicality_band"],
            f"{row['atypicality_zscore']:.2f}",
        ])

    detail_table = Table(detail_data, colWidths=[4*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2*cm])
    detail_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#37474f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(detail_table)
    story.append(Spacer(1, 0.5*cm))

    # Disclaimer
    story.append(Paragraph(
        "<i>Disclaimer: Atypicality scores are statistical deviation indices derived from training data "
        "distance metrics. They are not validated clinical risk assessments and should not be used for "
        "diagnostic or therapeutic decisions without independent experimental validation.</i>",
        styles["Italic"]
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# ──────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────
def _parse_fasta_text(content: str):
    if BIOPYTHON_AVAILABLE:
        handle = StringIO(content)
        records = list(SeqIO.parse(handle, "fasta"))
        parsed = []
        for idx, rec in enumerate(records, start=1):
            parsed.append({"id": rec.id or f"seq_{idx}", "sequence": str(rec.seq)})
        return parsed

    # Pure-Python fallback when BioPython is unavailable
    lines = content.splitlines()
    parsed = []
    current_id = None
    current_seq = []
    idx = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                parsed.append({"id": current_id, "sequence": "".join(current_seq)})
                idx += 1
            current_id = line[1:].split()[0] or f"seq_{idx + 1}"
            current_seq = []
        else:
            current_seq.append(line)
    if current_id is not None:
        parsed.append({"id": current_id, "sequence": "".join(current_seq)})
    return parsed


def _parse_plain_text(content: str):
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    return [{"id": f"seq_{i + 1}", "sequence": line} for i, line in enumerate(lines)]


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────
def _predict_rows(rows):
    sequences = [clean_sequence(row["sequence"]) for row in rows]
    outputs = predict_sequences(sequences)
    result_rows = []
    for row, pred in zip(rows, outputs):
        result_rows.append(
            {
                "id": row["id"],
                "sequence": clean_sequence(row["sequence"]),
                "input_length": len(row["sequence"]),
                "clean_length": pred["sequence_length"],
                "sequence_length": pred["sequence_length"],
                "predicted_virus": pred["predicted_virus"],
                "confidence": round(pred["confidence"], 4),
                "ebola_probability": round(pred["ebola_probability"], 4),
                "atypicality_index": round(pred["atypicality_index"], 2),
                "atypicality_band": pred["atypicality_band"],
                "atypicality_zscore": round(pred["atypicality_zscore"], 3),
                "mutation_risk_score": round(pred.get("mutation_risk_score", 0), 2),
                "esm_unavailable": pred.get("esm_unavailable", False),
            }
        )
    result_df = pd.DataFrame(result_rows)
    return result_df


# ──────────────────────────────────────────────
# Summary dashboard
# ──────────────────────────────────────────────
def _render_summary_dashboard(result_df: pd.DataFrame):
    st.markdown("---")
    st.subheader("Batch Summary Dashboard")

    # KPI cards with color coding
    metrics = [
        ("Sequences", len(result_df), "#f5f5f5", "#333"),
        ("Ebola", (result_df["predicted_virus"] == "Ebola").sum(), "#ffebee", "#c62828"),
        ("Lassa", (result_df["predicted_virus"] == "Lassa").sum(), "#e3f2fd", "#1565c0"),
        ("Unknown", (result_df["predicted_virus"].str.contains("Unknown")).sum(), "#fff3e0", "#e65100"),
        ("Mean Atypicality", f"{result_df['atypicality_index'].mean():.1f}", "#f3e5f5", "#6a1b9a"),
        ("Mean Comp Deviation", f"{result_df.get('mutation_risk_score', pd.Series([0]*len(result_df))).mean():.1f}", "#e8f5e9", "#2e7d32"),
    ]
    cols = st.columns(len(metrics))
    for col, (label, val, bg, fg) in zip(cols, metrics):
        col.markdown(
            f"""
            <div style="text-align:center; padding:12px 6px; background:{bg}; border-radius:10px; border:1px solid {fg}30;">
                <div style="font-size:11px; color:#666; margin-bottom:4px;">{label}</div>
                <div style="font-size:24px; font-weight:bold; color:{fg};">{val}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row 1
    if MATPLOTLIB_AVAILABLE:
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(_fig_class_distribution(result_df), use_container_width=True)
        with c2:
            st.pyplot(_fig_band_distribution(result_df), use_container_width=True)

        # Charts row 2
        c3, c4 = st.columns(2)
        with c3:
            st.pyplot(_fig_confidence_vs_atypicality(result_df), use_container_width=True)
        with c4:
            st.pyplot(_fig_atypicality_per_sequence(result_df), use_container_width=True)
    else:
        st.info("Install `matplotlib` to enable charts.")


# ──────────────────────────────────────────────
# Export
# ──────────────────────────────────────────────
def _render_report_download(result_df: pd.DataFrame):
    st.markdown("---")
    st.subheader("Export Results")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download CSV",
            result_df.to_csv(index=False).encode("utf-8"),
            file_name="sequence_predictions.csv",
            mime="text/csv",
        )
    with c2:
        if REPORTLAB_AVAILABLE:
            pdf_bytes = _build_pdf_report(result_df)
            st.download_button(
                "Download PDF Report",
                pdf_bytes,
                file_name="sequence_report.pdf",
                mime="application/pdf",
            )
        else:
            st.info("Install `reportlab` to enable PDF export.")


# ──────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────
if not MODEL_PATH.exists():
    st.error("Model file not found. Please ensure the model is deployed with the app.")
else:
    if METRICS_PATH.exists():
        st.caption(f"Using trained model: `{MODEL_PATH.name}`")

    uploaded = st.file_uploader(
        "Upload FASTA (.fasta/.fa/.faa), CSV (must include 'sequence'), or TXT (one sequence per line)",
        type=["fasta", "fa", "faa", "csv", "txt"],
    )

    if uploaded is not None:
        suffix = Path(uploaded.name).suffix.lower()
        content = uploaded.read().decode("utf-8", errors="ignore")

        try:
            if suffix in {".fasta", ".fa", ".faa"}:
                rows = _parse_fasta_text(content)
            elif suffix == ".csv":
                df = pd.read_csv(StringIO(content))
                if "sequence" not in df.columns:
                    st.error("CSV must contain a `sequence` column.")
                    st.stop()
                id_col = "id" if "id" in df.columns else None
                rows = [
                    {
                        "id": str(df.iloc[i][id_col]) if id_col else f"seq_{i + 1}",
                        "sequence": str(df.iloc[i]["sequence"]),
                    }
                    for i in range(len(df))
                ]
            else:
                rows = _parse_plain_text(content)

            if not rows:
                st.error("No valid sequences found in the uploaded file.")
                st.stop()

            result_df = _predict_rows(rows)
            st.success(f"Predicted {len(result_df)} sequence(s).")

            # Warn if ESM-2 embeddings are unavailable (torch not installed)
            if result_df.get("esm_unavailable", pd.Series([False])).any():
                st.warning(
                    "ESM-2 embeddings unavailable (torch not installed). "
                    "Running in fallback mode with composition-only features. "
                    "Accuracy may be reduced for atypical sequences."
                )

            st.dataframe(result_df.drop(columns=["sequence"]), use_container_width=True)

            # Dashboard
            _render_summary_dashboard(result_df)

            # Report cards
            if len(result_df) == 1:
                _render_report_card(result_df.iloc[0].to_dict())
            else:
                st.markdown("---")
                st.subheader("Individual Sequence Reports")
                selected_id = st.selectbox("Select a sequence for detailed report", result_df["id"].tolist())
                selected_row = result_df[result_df["id"] == selected_id].iloc[0].to_dict()
                _render_report_card(selected_row)

            # Export
            _render_report_download(result_df)

        except Exception as exc:
            st.error(f"Failed to process file: {exc}")

