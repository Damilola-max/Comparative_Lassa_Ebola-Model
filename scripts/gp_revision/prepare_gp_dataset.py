"""Prepare new GP-only dataset: EBOV GP (all outbreaks) + LASV GP."""
from Bio import SeqIO
import os, pandas as pd

base = "/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model"

# --- EBOV GP: combine Makona + non-Makona ---
print("Loading EBOV GP sequences...")
makona = list(SeqIO.parse(f"{base}/data/raw_all_outbreaks/ebov_makona_gp_only.fasta", "fasta"))
non_makona = list(SeqIO.parse(f"{base}/data/raw_all_outbreaks/ebov_non_makona_gp_only.fasta", "fasta"))

# Trim Makona GP to reference length (first 676 aa)
ref_gp = str(SeqIO.read(f"{base}/data/Reference/Ebola_Reference_Sequence.fasta", "fasta").seq)
ref_len = len(ref_gp)
print(f"Reference GP length: {ref_len}")

e_records = []
for r in makona:
    seq = str(r.seq)[:ref_len]
    e_records.append((r.id, "Ebola", seq))
for r in non_makona:
    seq = str(r.seq)
    # Trim/pad to reference length
    if len(seq) >= ref_len:
        seq = seq[:ref_len]
    e_records.append((r.id, "Ebola", seq))

print(f"Total EBOV GP: {len(e_records)}")

# --- LASV GP: use existing S_protein ---
print("Loading LASV GP sequences...")
lasv = list(SeqIO.parse(f"{base}/data/raw/S_protein.fas", "fasta"))
l_records = []
for r in lasv:
    l_records.append((r.id, "Lassa", str(r.seq)))
print(f"Total LASV GP: {len(l_records)}")

# --- Clean using same rules as original ---
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

def clean_sequence(seq):
    return "".join(c for c in seq.upper() if c in AMINO_ACIDS)

all_records = []
for sid, virus, seq in e_records + l_records:
    cleaned = clean_sequence(seq)
    if len(cleaned) >= 10:
        all_records.append({
            "id": sid,
            "virus": virus,
            "sequence": cleaned,
            "length": len(cleaned),
            "n_unknown": len(seq) - len(cleaned)
        })

df = pd.DataFrame(all_records)
out_path = f"{base}/data/cleaned/cleaned_sequences_gp_only.csv"
os.makedirs(f"{base}/data/cleaned", exist_ok=True)
df.to_csv(out_path, index=False)

print(f"\nCleaned dataset saved to {out_path}")
print(df.groupby("virus")["length"].agg(["count", "mean", "min", "max"]).round(1))
