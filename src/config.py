from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "cleaned" / "cleaned_sequences_gp_only.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "gp_revision"
MODEL_PATH = MODEL_DIR / "gp_classifier_v2.joblib"
METRICS_PATH = MODEL_DIR / "training_metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
