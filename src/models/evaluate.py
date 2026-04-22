import json

from src.config import METRICS_PATH


def load_training_metrics() -> dict:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Metrics file not found at {METRICS_PATH}. Train a model first."
        )
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    print(json.dumps(load_training_metrics(), indent=2))
