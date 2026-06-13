from typing import Iterable, List

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_PATH
from src.features.sequence_features import amino_acid_frequency_features, clean_sequence


def load_model_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train first with: python scripts/03_train.py"
        )
    saved = joblib.load(MODEL_PATH)

    if isinstance(saved, dict) and "model" in saved:
        return saved

    return {
        "model": saved,
        "risk_calibration": None,
        "feature_columns": None,
    }


def _atypicality_band_from_score(score: float) -> str:
    if score < 20:
        return "Low"
    if score < 40:
        return "Below-Average"
    if score < 60:
        return "Average"
    if score < 80:
        return "Elevated"
    return "High"


def _compute_atypicality_scores(features: pd.DataFrame, preds: np.ndarray, calibration: dict) -> List[dict]:
    if not calibration:
        return [{"atypicality_index": 50.0, "atypicality_zscore": 0.0}] * len(features)

    feature_columns = calibration["feature_columns"]
    aligned = features.reindex(columns=feature_columns, fill_value=0.0)

    mean = np.array(calibration["scaler_mean"], dtype=float)
    scale = np.array(calibration["scaler_scale"], dtype=float)
    scale = np.where(scale == 0, 1.0, scale)

    X_scaled = (aligned.values - mean) / scale
    out_scores: List[dict] = []

    for row, pred in zip(X_scaled, preds):
        key = str(int(pred))
        centroid = np.array(calibration["class_centroids"][key], dtype=float)
        stats = calibration["class_distance_stats"][key]
        dist = np.linalg.norm(row - centroid)
        z = (dist - stats["mean"]) / max(stats["std"], 1e-8)
        atyp_index = 50.0 + 15.0 * z
        atyp_index = max(0.0, min(100.0, atyp_index))
        out_scores.append(
            {
                "atypicality_index": float(atyp_index),
                "atypicality_zscore": float(z),
            }
        )

    return out_scores


def predict_sequences(sequences: Iterable[str]) -> List[dict]:
    bundle = load_model_bundle()
    model = bundle["model"]
    cleaned = [clean_sequence(s) for s in sequences]
    features = amino_acid_frequency_features(cleaned)

    probs = model.predict_proba(features)[:, 1]
    preds = (probs >= 0.5).astype(int)
    atypicality_scores = _compute_atypicality_scores(features, preds, bundle.get("risk_calibration"))

    output = []
    for sequence, prob, pred, atyp_info in zip(cleaned, probs, preds, atypicality_scores):
        label = "Ebola" if pred == 1 else "Lassa"
        confidence = prob if pred == 1 else (1 - prob)
        atyp_index = atyp_info["atypicality_index"]
        output.append(
            {
                "sequence_length": len(sequence),
                "predicted_virus": label,
                "confidence": float(confidence),
                "ebola_probability": float(prob),
                "atypicality_index": float(atyp_index),
                "atypicality_band": _atypicality_band_from_score(float(atyp_index)),
                "atypicality_zscore": float(atyp_info["atypicality_zscore"]),
            }
        )
    return output


def predict_from_dataframe(df: pd.DataFrame, sequence_column: str = "sequence") -> pd.DataFrame:
    predictions = predict_sequences(df[sequence_column].astype(str).tolist())
    result_df = pd.DataFrame(predictions)
    return pd.concat([df.reset_index(drop=True), result_df], axis=1)
