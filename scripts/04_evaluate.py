import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.evaluate import load_training_metrics


if __name__ == "__main__":
    metrics = load_training_metrics()
    print(json.dumps(metrics, indent=2))
