import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DATA_PATH, METRICS_PATH, MODEL_DIR, MODEL_PATH, RANDOM_STATE, TEST_SIZE
from src.features.sequence_features import build_training_frame


def _build_candidate_models() -> dict:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=300,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }


def _build_risk_calibration(model, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    class_centroids = {}
    class_distance_stats = {}

    for class_idx, class_name in [(0, "Lassa"), (1, "Ebola")]:
        class_mask = y_train.values == class_idx
        class_points = X_scaled[class_mask]
        centroid = class_points.mean(axis=0)
        distances = np.linalg.norm(class_points - centroid, axis=1)

        class_centroids[str(class_idx)] = centroid.tolist()
        class_distance_stats[str(class_idx)] = {
            "mean": float(distances.mean()),
            "std": float(distances.std() if distances.std() > 1e-8 else 1e-8),
            "class_name": class_name,
        }

    return {
        "feature_columns": X_train.columns.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "class_centroids": class_centroids,
        "class_distance_stats": class_distance_stats,
    }


def train_best_model() -> dict:
    raw = pd.read_csv(DATA_PATH)
    raw = raw[["sequence", "virus"]].dropna().copy()

    frame = build_training_frame(raw)
    X = frame.drop(columns=["virus", "sequence"])
    y = frame["virus"].str.lower().map({"lassa": 0, "ebola": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    best_model_name = ""
    best_model = None
    best_f1 = -1.0
    leaderboard = {}

    for name, model in _build_candidate_models().items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds)),
            "recall": float(recall_score(y_test, preds)),
            "f1": float(f1_score(y_test, preds)),
            "roc_auc": float(roc_auc_score(y_test, probs)),
        }
        leaderboard[name] = metrics

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model_name = name
            best_model = model

    risk_calibration = _build_risk_calibration(best_model, X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": best_model,
        "risk_calibration": risk_calibration,
        "feature_columns": X_train.columns.tolist(),
    }
    joblib.dump(bundle, MODEL_PATH)

    output = {
        "best_model": best_model_name,
        "leaderboard": leaderboard,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    METRICS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    result = train_best_model()
    print(json.dumps(result, indent=2))
