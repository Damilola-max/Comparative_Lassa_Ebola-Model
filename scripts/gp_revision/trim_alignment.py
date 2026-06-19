"""Trim LASV alignment to core positions with >50% coverage."""
from Bio import SeqIO
import numpy as np
from collections import Counter

base = "/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model"

# Load aligned sequences
lasv_aligned = [str(r.seq) for r in SeqIO.parse(f"{base}/results/gp_revision/lasv_gp_sample.aligned.fasta", "fasta")]

# Find positions with >50% non-gap coverage
core_positions = []
for pos in range(len(lasv_aligned[0])):
    chars = [s[pos] for s in lasv_aligned if s[pos] != "-"]
    coverage = len(chars) / len(lasv_aligned)
    if coverage > 0.5:
        core_positions.append(pos)

print(f"LASV: {len(lasv_aligned[0])} aligned positions -> {len(core_positions)} core positions (>50% coverage)")

# Recompute entropy on core positions only
entropy = []
for pos in core_positions:
    chars = [s[pos] for s in lasv_aligned if s[pos] != "-"]
    counts = Counter(chars)
    total = len(chars)
    probs = [c/total for c in counts.values()]
    H = -sum(p * np.log2(p) for p in probs if p > 0)
    entropy.append(H)

def classify_sites(entropy):
    return ["Critical" if H < 0.1 else "Conserved" if H < 0.5 else "Intermediate" if H < 1.0 else "Hotspot" for H in entropy]

lasv_cat = classify_sites(entropy)
print("\nLASV GP (core positions):")
for cat in ["Critical", "Conserved", "Intermediate", "Hotspot"]:
    n = lasv_cat.count(cat)
    print(f"  {cat}: {n} ({n/len(lasv_cat)*100:.1f}%)")

# Also do EBOV core positions
ebov_aligned = [str(r.seq) for r in SeqIO.parse(f"{base}/results/gp_revision/ebov_gp_sample.aligned.fasta", "fasta")]
ebov_core = [pos for pos in range(len(ebov_aligned[0])) if sum(1 for s in ebov_aligned if s[pos] != "-") / len(ebov_aligned) > 0.5]
print(f"\nEBOV: {len(ebov_aligned[0])} aligned positions -> {len(ebov_core)} core positions")

ebov_entropy = []
for pos in ebov_core:
    chars = [s[pos] for s in ebov_aligned if s[pos] != "-"]
    counts = Counter(chars)
    total = len(chars)
    probs = [c/total for c in counts.values()]
    H = -sum(p * np.log2(p) for p in probs if p > 0)
    ebov_entropy.append(H)

ebov_cat = classify_sites(ebov_entropy)
print("\nEBOV GP (core positions):")
for cat in ["Critical", "Conserved", "Intermediate", "Hotspot"]:
    n = ebov_cat.count(cat)
    print(f"  {cat}: {n} ({n/len(ebov_cat)*100:.1f}%)")
