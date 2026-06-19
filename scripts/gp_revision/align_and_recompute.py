"""Align GP sequences with MAFFT and recompute site categories."""
import subprocess, os, random
from Bio import SeqIO
from collections import Counter
import numpy as np
import pandas as pd

base = "/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model"
out_dir = f"{base}/results/gp_revision"

# Load cleaned sequences
df = pd.read_csv(f"{base}/data/cleaned/cleaned_sequences_gp_only.csv")
df = df[df["length"] >= 300].copy()

ebola_seqs = df[df["virus"] == "Ebola"]["sequence"].tolist()
lassa_seqs = df[df["virus"] == "Lassa"]["sequence"].tolist()

random.seed(42)
ebola_sample = random.sample(ebola_seqs, min(200, len(ebola_seqs)))
lassa_sample = random.sample(lassa_seqs, min(200, len(lassa_seqs)))

print(f"EBOV sample: {len(ebola_sample)}, LASV sample: {len(lassa_sample)}")

# Write subsets to FASTA
def write_fasta(seqs, path, prefix):
    with open(path, "w") as f:
        for i, seq in enumerate(seqs):
            f.write(f">{prefix}_{i}\n{seq}\n")

ebov_fasta = f"{out_dir}/ebov_gp_sample.fasta"
lasv_fasta = f"{out_dir}/lasv_gp_sample.fasta"
write_fasta(ebola_sample, ebov_fasta, "EBOV")
write_fasta(lassa_sample, lasv_fasta, "LASV")

# Run MAFFT
print("\nAligning EBOV GP with MAFFT...")
subprocess.run(["mafft", "--auto", ebov_fasta], stdout=open(f"{out_dir}/ebov_gp_sample.aligned.fasta", "w"), check=True)
print("Aligning LASV GP with MAFFT...")
subprocess.run(["mafft", "--auto", lasv_fasta], stdout=open(f"{out_dir}/lasv_gp_sample.aligned.fasta", "w"), check=True)

# Compute entropy on aligned sequences
def compute_aligned_entropy(aligned_fasta, max_pos=800):
    seqs = [str(r.seq) for r in SeqIO.parse(aligned_fasta, "fasta")]
    # Trim terminal gaps
    min_len = min(len(s) for s in seqs)
    
    entropy = []
    for pos in range(min_len):
        chars = [s[pos] for s in seqs if s[pos] != "-"]
        if not chars:
            continue
        counts = Counter(chars)
        total = len(chars)
        probs = [c/total for c in counts.values()]
        H = -sum(p * np.log2(p) for p in probs if p > 0)
        entropy.append(H)
    return np.array(entropy)

ebola_ent = compute_aligned_entropy(f"{out_dir}/ebov_gp_sample.aligned.fasta")
lassa_ent = compute_aligned_entropy(f"{out_dir}/lasv_gp_sample.aligned.fasta")

print(f"\nEBOV aligned positions: {len(ebola_ent)}")
print(f"LASV aligned positions: {len(lassa_ent)}")

def classify_sites(entropy):
    categories = []
    for H in entropy:
        if H < 0.1:
            categories.append("Critical")
        elif H < 0.5:
            categories.append("Conserved")
        elif H < 1.0:
            categories.append("Intermediate")
        else:
            categories.append("Hotspot")
    return categories

ebola_cat = classify_sites(ebola_ent)
lassa_cat = classify_sites(lassa_ent)

print("\n=== ALIGNED SITE CATEGORIES ===")
print("EBOV GP:")
for cat in ["Critical", "Conserved", "Intermediate", "Hotspot"]:
    n = ebola_cat.count(cat)
    print(f"  {cat}: {n} ({n/len(ebola_cat)*100:.1f}%)")

print("LASV GP:")
for cat in ["Critical", "Conserved", "Intermediate", "Hotspot"]:
    n = lassa_cat.count(cat)
    print(f"  {cat}: {n} ({n/len(lassa_cat)*100:.1f}%)")

# Save
site_df = pd.DataFrame({
    "virus": ["Ebola"] * len(ebola_cat) + ["Lassa"] * len(lassa_cat),
    "position": list(range(len(ebola_cat))) + list(range(len(lassa_cat))),
    "entropy": list(ebola_ent) + list(lassa_ent),
    "category": ebola_cat + lassa_cat
})
site_df.to_csv(f"{out_dir}/site_categories_aligned.csv", index=False)

# Generate new figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig_dir = f"{out_dir}/figures"
cats = ["Critical", "Conserved", "Intermediate", "Hotspot"]
ebola_pcts = [ebola_cat.count(c) / len(ebola_cat) * 100 for c in cats]
lassa_pcts = [lassa_cat.count(c) / len(lassa_cat) * 100 for c in cats]

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(cats))
width = 0.35
ax.bar(x - width/2, ebola_pcts, width, label="EBOV GP", color="#d62728")
ax.bar(x + width/2, lassa_pcts, width, label="LASV GP", color="#1f77b4")
ax.set_ylabel("Fraction (%)")
ax.set_title("Site Category Distribution: GP vs GP (MAFFT-aligned)")
ax.set_xticks(x)
ax.set_xticklabels(cats)
ax.legend()
plt.tight_layout()
plt.savefig(f"{fig_dir}/site_category_comparison_gp_aligned.png", dpi=150)
plt.close()
print(f"Saved: {fig_dir}/site_category_comparison_gp_aligned.png")

# Entropy profile
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(ebola_ent, label="EBOV GP", color="#d62728", alpha=0.8)
ax.plot(lassa_ent, label="LASV GP", color="#1f77b4", alpha=0.8)
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Aligned Position")
ax.set_ylabel("Shannon Entropy")
ax.set_title("Per-Position Entropy Profile (MAFFT-aligned GP)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{fig_dir}/entropy_profile_gp_aligned.png", dpi=150)
plt.close()
print(f"Saved: {fig_dir}/entropy_profile_gp_aligned.png")

print("\n=== DONE ===")
