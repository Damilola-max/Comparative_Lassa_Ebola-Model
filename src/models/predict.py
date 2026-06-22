from typing import Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd
from src.config import MODEL_PATH
from src.features.sequence_features import amino_acid_frequency_features, clean_sequence


_ESM_CACHE = {}


def _load_esm_model():
    if "model" in _ESM_CACHE:
        return _ESM_CACHE["model"], _ESM_CACHE["alphabet"]
    try:
        import torch
        import esm
        model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t12_35M_UR50D")
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        _ESM_CACHE["model"] = model
        _ESM_CACHE["alphabet"] = alphabet
        return model, alphabet
    except ImportError:
        raise RuntimeError("ESM-2 requires 'torch' and 'fair-esm'. Install: pip install torch fair-esm")


def _embed_sequences(sequences: List[str]) -> np.ndarray:
    import torch
    model, alphabet = _load_esm_model()
    batch_converter = alphabet.get_batch_converter()
    data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
    _, _, tokens = batch_converter(data)
    if torch.cuda.is_available():
        tokens = tokens.cuda()
    with torch.no_grad():
        results = model(tokens, repr_layers=[12], return_contacts=False)
    token_representations = results["representations"][12]
    embeddings = []
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        emb = token_representations[i, 1:seq_len + 1].mean(dim=0).cpu().numpy()
        embeddings.append(emb)
    return np.array(embeddings)


def load_model_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    saved = joblib.load(MODEL_PATH)
    if isinstance(saved, dict) and "model" in saved:
        return saved
    return {"model": saved, "scaler": None, "risk_calibration": None, "feature_columns": None, "esm_dim": 0}


_BUNDLE_CACHE = {}


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


def _compute_atypicality_scores(features: pd.DataFrame, preds: np.ndarray, calibration: Optional[dict]) -> List[dict]:
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
        out_scores.append({"atypicality_index": float(atyp_index), "atypicality_zscore": float(z)})
    return out_scores


def predict_sequences(sequences: Iterable[str]) -> List[dict]:
    if not _BUNDLE_CACHE:
        _BUNDLE_CACHE.update(load_model_bundle())
    model = _BUNDLE_CACHE["model"]
    calibration = _BUNDLE_CACHE.get("risk_calibration")
    feature_columns = _BUNDLE_CACHE.get("feature_columns")
    cleaned = [clean_sequence(s) for s in sequences]
    comp_features = amino_acid_frequency_features(cleaned)
    # Align columns to training order if available
    if feature_columns:
        comp_features = comp_features.reindex(columns=feature_columns, fill_value=0.0)
    # Pipeline handles scaling internally — pass DataFrame directly
    probs = model.predict_proba(comp_features)[:, 1]
    preds = (probs >= 0.5).astype(int)
    atypicality_scores = _compute_atypicality_scores(comp_features, preds, calibration)
    esm_unavailable = False
    output = []
    for sequence, prob, pred, atyp_info in zip(cleaned, probs, preds, atypicality_scores):
        atyp_index = atyp_info["atypicality_index"]
        atyp_z = atyp_info["atypicality_zscore"]

        # Outlier detection: if atypicality is extremely high, flag as Unknown
        if atyp_index >= 95 or atyp_z >= 3.0:
            label = "Unknown / Highly Atypical"
            confidence = float(max(prob, 1 - prob))
        else:
            label = "Ebola" if pred == 1 else "Lassa"
            confidence = prob if pred == 1 else (1 - prob)

        # Mutation risk score (0-100): how far from training centroid
        mutation_risk = min(100.0, max(0.0, atyp_index * 1.2))

        output.append({
            "sequence_length": len(sequence),
            "predicted_virus": label,
            "confidence": float(confidence),
            "ebola_probability": float(prob),
            "atypicality_index": float(atyp_index),
            "atypicality_band": _atypicality_band_from_score(float(atyp_index)),
            "atypicality_zscore": float(atyp_z),
            "mutation_risk_score": float(mutation_risk),
            "esm_unavailable": esm_unavailable,
        })
    return output


def predict_from_dataframe(df: pd.DataFrame, sequence_column: str = "sequence") -> pd.DataFrame:
    predictions = predict_sequences(df[sequence_column].astype(str).tolist())
    result_df = pd.DataFrame(predictions)
    return pd.concat([df.reset_index(drop=True), result_df], axis=1)
