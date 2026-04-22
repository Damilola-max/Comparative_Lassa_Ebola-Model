import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.train import train_best_model


if __name__ == "__main__":
    metrics = train_best_model()
    print(json.dumps(metrics, indent=2))
