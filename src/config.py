from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "cleaned" / "cleaned_sequences.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "final"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
METRICS_PATH = MODEL_DIR / "training_metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
