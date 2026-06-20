"""Add calibration data to GP revision model so atypicality scoring works in app."""
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

base = "/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model"

# Load model
model = joblib.load(f"{base}/models/gp_revision/gp_classifier.joblib")
print("Model loaded.")

# Load training data
df = pd.read_csv(f"{base}/data/cleaned/cleaned_sequences_gp_only.csv")

from src.features.sequence_features import amino_acid_frequency_features
cleaned = df["sequence"].apply(lambda s: "".join(c for c in s.upper() if c in "ACDEFGHIKLMNPQRSTVWY"))
features = amino_acid_frequency_features(cleaned.tolist())

# Predict
probs = model.predict_proba(features)[:, 1]
preds = (probs >= 0.5).astype(int)

# Compute class centroids and distance stats
feature_columns = list(features.columns)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features.values)

class_centroids = {}
class_distance_stats = {}

for label in [0, 1]:
    mask = preds == label
    class_samples = X_scaled[mask]
    centroid = class_samples.mean(axis=0)
    dists = np.linalg.norm(class_samples - centroid, axis=1)
    class_centroids[str(label)] = centroid.tolist()
    class_distance_stats[str(label)] = {"mean": float(dists.mean()), "std": float(dists.std())}

calibration = {
    "feature_columns": feature_columns,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "class_centroids": class_centroids,
    "class_distance_stats": class_distance_stats,
}

bundle = {
    "model": model,
    "risk_calibration": calibration,
    "feature_columns": feature_columns,
}

joblib.dump(bundle, f"{base}/models/gp_revision/gp_classifier.joblib")
print("Model saved with calibration.")
print(f"Class 0 (Lassa): centroid distance mean={class_distance_stats['0']['mean']:.4f}, std={class_distance_stats['0']['std']:.4f}")
print(f"Class 1 (Ebola): centroid distance mean={class_distance_stats['1']['mean']:.4f}, std={class_distance_stats['1']['std']:.4f}")
