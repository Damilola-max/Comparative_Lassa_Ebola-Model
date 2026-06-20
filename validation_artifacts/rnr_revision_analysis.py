import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.features.sequence_features import amino_acid_frequency_features  # noqa: E402
from src.models.predict import predict_sequences  # noqa: E402

OUT_DIR = PROJECT_ROOT / "results" / "rnr_revision"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

for p in [OUT_DIR, FIG_DIR, TABLE_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def parse_seq_id(seq_id: str, virus: str) -> Dict[str, str]:
    parts = str(seq_id).split("|")
    if virus.lower() == "lassa":
        lineage = parts[2] if len(parts) > 2 else "unknown"
        return {
            "protein_scope": "LASV_S_protein",
            "segment": parts[1] if len(parts) > 1 else "s",
            "lineage": lineage,
            "host": "unknown",
            "country": "unknown",
            "collection_date": "unknown",
            "group_key": f"LASV_{lineage}",
        }

    country = parts[3] if len(parts) > 3 else "unknown"
    date = parts[5] if len(parts) > 5 else "unknown"
    year = date[:4] if isinstance(date, str) and len(date) >= 4 and date[:4].isdigit() else "unknown"
    return {
        "protein_scope": "EBOV_Makona_polyprotein",
        "segment": "polyprotein",
        "lineage": "Makona_context",
        "host": "human",
        "country": country,
        "collection_date": date,
        "group_key": f"EBOV_{country}_{year}",
    }


def ci95(values: List[float]) -> Tuple[float, float, float]:
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, mean, mean
    se = arr.std(ddof=1) / np.sqrt(len(arr))
    return mean, float(mean - 1.96 * se), float(mean + 1.96 * se)


def model_registry(random_state: int = 42) -> Dict[str, object]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, random_state=random_state)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }


def evaluate_with_splitter(X: np.ndarray, y: np.ndarray, splitter, groups=None) -> pd.DataFrame:
    rows = []
    for fold_idx, (tr, te) in enumerate(splitter.split(X, y, groups), start=1):
        X_train, X_test = X[tr], X[te]
        y_train, y_test = y[tr], y[te]

        for model_name, model in model_registry().items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

            if len(np.unique(y_test)) > 1:
                roc_auc = roc_auc_score(y_test, probs)
            else:
                roc_auc = np.nan

            rows.append(
                {
                    "fold": fold_idx,
                    "model": model_name,
                    "accuracy": accuracy_score(y_test, preds),
                    "precision": precision_score(y_test, preds, zero_division=0),
                    "recall": recall_score(y_test, preds, zero_division=0),
                    "f1": f1_score(y_test, preds, zero_division=0),
                    "roc_auc": roc_auc,
                    "n_train": len(tr),
                    "n_test": len(te),
                }
            )

    return pd.DataFrame(rows)


def summarize_metrics(df: pd.DataFrame, eval_name: str) -> pd.DataFrame:
    out = []
    for model_name, sub in df.groupby("model"):
        rec = {"evaluation": eval_name, "model": model_name, "n_folds": int(sub.shape[0])}
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            mean, lo, hi = ci95(sub[metric].tolist())
            rec[f"{metric}_mean"] = mean
            rec[f"{metric}_ci95_low"] = lo
            rec[f"{metric}_ci95_high"] = hi
        out.append(rec)
    return pd.DataFrame(out)


def to_png_table(df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(14, max(2.5, 0.35 * len(df) + 1.5)))
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    cleaned_path = PROJECT_ROOT / "data" / "cleaned" / "cleaned_sequences.csv"
    emb_tensor_path = PROJECT_ROOT / "notebooks" / "embedding" / "all_embeddings_COMPLETE.pt"
    emb_meta_path = PROJECT_ROOT / "notebooks" / "embedding" / "all_metadata.csv"

    df = pd.read_csv(cleaned_path)
    df = df.rename(columns={"id": "sequence_id"})
    df["length"] = df["sequence"].str.len()

    parsed = df.apply(lambda r: parse_seq_id(r["sequence_id"], r["virus"]), axis=1, result_type="expand")
    manifest = pd.concat([df[["sequence_id", "virus", "sequence", "length"]], parsed], axis=1)
    manifest.to_csv(TABLE_DIR / "dataset_manifest.csv", index=False)

    virus_summary = (
        manifest.groupby("virus")
        .agg(n_sequences=("sequence_id", "count"), mean_length=("length", "mean"), median_length=("length", "median"))
        .reset_index()
    )
    virus_summary.to_csv(TABLE_DIR / "dataset_virus_summary.csv", index=False)

    missingness = manifest[["lineage", "country", "collection_date", "host", "protein_scope"]].isna().mean().reset_index()
    missingness.columns = ["field", "missing_fraction"]
    missingness.to_csv(TABLE_DIR / "dataset_missingness.csv", index=False)

    exact_dup_total = int(manifest.duplicated(subset=["sequence"]).sum())
    exact_dup_within = (
        manifest.groupby("virus")["sequence"]
        .apply(lambda s: int(s.duplicated().sum()))
        .reset_index(name="exact_duplicates_within_virus")
    )
    exact_dup_within["exact_duplicates_global"] = exact_dup_total
    exact_dup_within.to_csv(TABLE_DIR / "duplicate_summary.csv", index=False)

    # Composition features
    X_comp_df = amino_acid_frequency_features(manifest["sequence"].tolist())
    y = manifest["virus"].str.lower().map({"lassa": 0, "ebola": 1}).values
    groups = manifest["group_key"].astype(str).values

    # Embeddings
    emb_tensor = torch.load(emb_tensor_path, map_location="cpu")
    emb_meta = pd.read_csv(emb_meta_path)
    emb_meta = emb_meta.rename(columns={"id": "sequence_id"})
    emb_index = emb_meta[["sequence_id", "embedding_idx"]].drop_duplicates()

    merged = manifest[["sequence_id"]].merge(emb_index, on="sequence_id", how="left")
    if merged["embedding_idx"].isna().any():
        raise ValueError("Some sequence IDs are missing embedding indices.")

    emb_idx = merged["embedding_idx"].astype(int).values
    X_esm = emb_tensor[emb_idx].numpy()

    X_comp = X_comp_df.values
    X_combined = np.hstack([X_comp, X_esm])

    # Repeated stratified CV (composition baseline)
    rs_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    comp_cv_raw = evaluate_with_splitter(X_comp, y, rs_cv)
    comp_cv_raw.to_csv(TABLE_DIR / "cv_composition_raw.csv", index=False)

    # Group CV (leakage control)
    group_cv = GroupKFold(n_splits=5)
    group_cv_raw = evaluate_with_splitter(X_comp, y, group_cv, groups=groups)
    group_cv_raw.to_csv(TABLE_DIR / "cv_grouped_raw.csv", index=False)

    # Ablation (same splitter type for fair comparison)
    ablation_rows = []
    for feature_name, X_feat in [
        ("composition_only", X_comp),
        ("esm_only", X_esm),
        ("composition_plus_esm", X_combined),
    ]:
        raw = evaluate_with_splitter(X_feat, y, rs_cv)
        raw["feature_set"] = feature_name
        raw.to_csv(TABLE_DIR / f"ablation_raw_{feature_name}.csv", index=False)
        ablation_rows.append(raw)

    ablation_all = pd.concat(ablation_rows, ignore_index=True)

    # External holdout by grouped split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X_comp, y, groups=groups))
    ext_rows = []
    for name, model in model_registry().items():
        model.fit(X_comp[tr_idx], y[tr_idx])
        preds = model.predict(X_comp[te_idx])
        probs = model.predict_proba(X_comp[te_idx])[:, 1]
        ext_rows.append(
            {
                "evaluation": "external_group_holdout",
                "model": name,
                "n_train": int(len(tr_idx)),
                "n_test": int(len(te_idx)),
                "accuracy": accuracy_score(y[te_idx], preds),
                "precision": precision_score(y[te_idx], preds, zero_division=0),
                "recall": recall_score(y[te_idx], preds, zero_division=0),
                "f1": f1_score(y[te_idx], preds, zero_division=0),
                "roc_auc": roc_auc_score(y[te_idx], probs),
            }
        )
    ext_df = pd.DataFrame(ext_rows)
    ext_df.to_csv(TABLE_DIR / "external_group_holdout_metrics.csv", index=False)

    # Summary tables
    comp_summary = summarize_metrics(comp_cv_raw, "repeated_stratified_cv_composition")
    group_summary = summarize_metrics(group_cv_raw, "group_kfold_composition")

    ablation_summary_records = []
    for (feature_set, model_name), sub in ablation_all.groupby(["feature_set", "model"]):
        rec = {"feature_set": feature_set, "model": model_name, "n_folds": int(sub.shape[0])}
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            mean, lo, hi = ci95(sub[metric].tolist())
            rec[f"{metric}_mean"] = mean
            rec[f"{metric}_ci95_low"] = lo
            rec[f"{metric}_ci95_high"] = hi
        ablation_summary_records.append(rec)
    ablation_summary = pd.DataFrame(ablation_summary_records)

    validation_summary = pd.concat([comp_summary, group_summary], ignore_index=True)
    validation_summary.to_csv(TABLE_DIR / "validation_summary.csv", index=False)
    ablation_summary.to_csv(TABLE_DIR / "ablation_summary.csv", index=False)

    # QA: canonical references + edge cases
    ref_rows = []
    for fasta_name in ["Lassa_Reference_Sequence.fasta", "Ebola_Reference_Sequence.fasta"]:
        fasta_path = PROJECT_ROOT / "data" / "Reference" / fasta_name
        records = list(SeqIO.parse(str(fasta_path), "fasta"))
        for rec in records:
            ref_rows.append({"id": rec.id, "sequence": str(rec.seq), "source": fasta_name})

    ref_preds = predict_sequences([r["sequence"] for r in ref_rows])
    ref_eval = pd.DataFrame(ref_rows)
    ref_eval = pd.concat([ref_eval, pd.DataFrame(ref_preds)], axis=1)
    ref_eval.to_csv(TABLE_DIR / "qa_reference_predictions.csv", index=False)

    edge_input = [
        {"id": "edge_short", "sequence": "ACD"},
        {"id": "edge_ambiguous", "sequence": "AXXZ---TT??"},
        {"id": "edge_numeric", "sequence": "A12345VVVV"},
        {"id": "edge_long_repeat", "sequence": "ACDEFGHIKLMNPQRSTVWY" * 20},
    ]
    edge_preds = predict_sequences([r["sequence"] for r in edge_input])
    edge_eval = pd.DataFrame(edge_input)
    edge_eval = pd.concat([edge_eval, pd.DataFrame(edge_preds)], axis=1)
    edge_eval.to_csv(TABLE_DIR / "qa_edge_case_predictions.csv", index=False)

    # Figures
    fig_data = ablation_summary[ablation_summary["model"] == "logistic_regression"].copy()
    fig_data = fig_data.sort_values("f1_mean")
    plt.figure(figsize=(8, 4.8))
    plt.barh(fig_data["feature_set"], fig_data["f1_mean"], color="#1976d2")
    plt.xlabel("F1 (mean across repeated stratified CV)")
    plt.title("Ablation: Logistic Regression Feature-Set Performance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ablation_f1_logistic.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8.5, 4.8))
    box_df = pd.concat(
        [
            comp_cv_raw.assign(eval_type="Repeated stratified CV"),
            group_cv_raw.assign(eval_type="Grouped CV"),
        ],
        ignore_index=True,
    )
    for i, eval_type in enumerate(["Repeated stratified CV", "Grouped CV"]):
        vals = box_df[(box_df["eval_type"] == eval_type) & (box_df["model"] == "logistic_regression")]["f1"].values
        x = np.random.normal(i + 1, 0.04, size=len(vals))
        plt.scatter(x, vals, alpha=0.6)
    plt.xticks([1, 2], ["Repeated stratified CV", "Grouped CV"])
    plt.ylabel("F1")
    plt.title("Validation Robustness: Logistic Regression F1 by Split Strategy")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "validation_split_comparison.png", dpi=220)
    plt.close()

    to_png_table(
        ref_eval[["id", "source", "predicted_virus", "confidence", "mutation_risk_score", "mutation_risk_category", "atypicality_zscore"]],
        FIG_DIR / "qa_reference_predictions.png",
        "QA Regression Panel: Canonical Reference Predictions",
    )
    to_png_table(
        edge_eval[["id", "predicted_virus", "confidence", "mutation_risk_score", "mutation_risk_category", "atypicality_zscore"]],
        FIG_DIR / "qa_edge_case_predictions.png",
        "QA Regression Panel: Edge-Case Input Predictions",
    )

    summary_payload = {
        "n_sequences": int(manifest.shape[0]),
        "n_lassa": int((manifest["virus"].str.lower() == "lassa").sum()),
        "n_ebola": int((manifest["virus"].str.lower() == "ebola").sum()),
        "exact_duplicate_sequences": exact_dup_total,
        "n_groups": int(pd.Series(groups).nunique()),
        "outputs": {
            "tables": str(TABLE_DIR),
            "figures": str(FIG_DIR),
        },
    }
    (OUT_DIR / "revision_analysis_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
