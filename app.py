from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st
from Bio import SeqIO

from src.config import METRICS_PATH, MODEL_PATH
from src.features.sequence_features import clean_sequence
from src.models.predict import predict_sequences


st.set_page_config(page_title="Lassa vs Ebola Sequence Classifier", layout="wide")
st.title("Comparative Lassa-Ebola Sequence Classifier")
st.write("Upload sequences and get a model prediction with confidence.")
st.caption(
    "Interpretation note: this model uses sequence-length and amino-acid composition features. "
    "Mutation risk is an outlier-style score based on distance from known class patterns in training data."
)


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
    return (
        f"Sequence {row['id']} was classified as {row['predicted_virus']} with {confidence_pct:.2f}% confidence "
        f"({ _confidence_band(row['confidence']) } confidence). "
        f"Its mutation risk score is {row['mutation_risk_score']:.2f}/100, which maps to the "
        f"'{row['mutation_risk_category']}' category. "
        f"The atypicality z-score is {row['atypicality_zscore']:.2f}, meaning this sequence is "
        f"{_atypicality_phrase(row['atypicality_zscore'])}. "
        "In practice, higher risk scores suggest the sequence pattern is less typical for its predicted class and may warrant closer review."
    )


def _parse_fasta_text(content: str):
    handle = StringIO(content)
    records = list(SeqIO.parse(handle, "fasta"))
    parsed = []
    for idx, rec in enumerate(records, start=1):
        parsed.append({"id": rec.id or f"seq_{idx}", "sequence": str(rec.seq)})
    return parsed


def _parse_plain_text(content: str):
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    return [{"id": f"seq_{i + 1}", "sequence": line} for i, line in enumerate(lines)]


def _predict_rows(rows):
    sequences = [clean_sequence(row["sequence"]) for row in rows]
    outputs = predict_sequences(sequences)
    result_rows = []
    for row, pred in zip(rows, outputs):
        result_rows.append(
            {
                "id": row["id"],
                "input_length": len(row["sequence"]),
                "clean_length": pred["sequence_length"],
                "predicted_virus": pred["predicted_virus"],
                "confidence": round(pred["confidence"], 4),
                "ebola_probability": round(pred["ebola_probability"], 4),
                "mutation_risk_score": round(pred["mutation_risk_score"], 2),
                "mutation_risk_category": pred["mutation_risk_category"],
                "atypicality_zscore": round(pred["atypicality_zscore"], 3),
            }
        )
    result_df = pd.DataFrame(result_rows)
    result_df["explanation"] = [
        _explain_prediction(row)
        for row in result_df.to_dict(orient="records")
    ]
    return result_df


def _render_summary_figures(result_df: pd.DataFrame):
    st.subheader("Prediction Summary")
    c1, c2 = st.columns(2)

    with c1:
        st.write("Predicted class counts")
        class_counts = result_df["predicted_virus"].value_counts().rename_axis("virus").reset_index(name="count")
        st.bar_chart(class_counts.set_index("virus"))

    with c2:
        st.write("Risk category counts")
        risk_counts = result_df["mutation_risk_category"].value_counts().rename_axis("risk").reset_index(name="count")
        st.bar_chart(risk_counts.set_index("risk"))

    st.write("Confidence and mutation risk per sequence")
    chart_df = result_df[["id", "confidence", "mutation_risk_score"]].copy()
    st.line_chart(chart_df.set_index("id"))


def _render_text_interpretation(result_df: pd.DataFrame):
    st.subheader("Detailed Interpretation")
    for row in result_df.to_dict(orient="records"):
        st.markdown(f"- {row['explanation']}")


if not MODEL_PATH.exists():
    st.warning("No trained model found yet. Run: `python scripts/03_train.py`")
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
            st.dataframe(result_df, use_container_width=True)
            _render_summary_figures(result_df)
            _render_text_interpretation(result_df)
            st.download_button(
                "Download predictions as CSV",
                result_df.to_csv(index=False).encode("utf-8"),
                file_name="sequence_predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Failed to process file: {exc}")
