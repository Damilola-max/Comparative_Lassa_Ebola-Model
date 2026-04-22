import re
from typing import Iterable, List

import pandas as pd


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(sequence: str) -> str:
    if not isinstance(sequence, str):
        return ""
    sequence = sequence.upper()
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", sequence)


def amino_acid_frequency_features(sequences: Iterable[str]) -> pd.DataFrame:
    rows: List[dict] = []
    for sequence in sequences:
        clean = clean_sequence(sequence)
        length = max(len(clean), 1)

        row = {"seq_length": len(clean)}
        for aa in AMINO_ACIDS:
            row[f"aa_freq_{aa}"] = clean.count(aa) / length
        rows.append(row)

    return pd.DataFrame(rows)


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    features = amino_acid_frequency_features(df["sequence"].tolist())
    features["virus"] = df["virus"].values
    features["sequence"] = df["sequence"].values
    return features
