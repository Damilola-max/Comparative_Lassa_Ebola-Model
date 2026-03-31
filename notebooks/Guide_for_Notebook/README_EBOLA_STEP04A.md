# Step 04B — Ebola Hybrid Mutation Scoring (Reference + Alignment + ESM-2)
**Notebook:** `04B — Ebola Hybrid Mutation Scoring (Reference + Alignment + ESM)-2.ipynb`  
**Goal:** For **Ebola**, compute mutation / site importance scores by combining:
- **Reference coordinate system** (so every site has a real reference position),
- **Multiple Sequence Alignment (MSA)** against the reference (MAFFT),
- **Per-site conservation / entropy** from the alignment,
- **ESM-2 embedding-based signals** (future sections can add embedding distances),
- **Substitution scoring** (BLOSUM62-based scoring for AA changes).

This notebook is a **per-virus Step 04** workflow. You will run the same style for Lassa and then compare results in later steps.

---

## 0) What you must already have before running this notebook

### Input files (Ebola)
These paths are used in Cell 0 / Cell 1:

- **Reference sequence** (single FASTA record):
  - `data/Reference/Ebola_Reference_Sequence.fasta`

- **Cleaned dataset FASTA** (Ebola sequences you want to analyze):
  - `data/processed/ebola_cleaned.fasta`

- **Ebola embeddings** (ESM-2 output, Ebola-only):
  - `data/embeddings/ebola_embeddings.pt`  (shape: `1610 × 1280`)

- **Ebola metadata** (Ebola-only; must align to embeddings row order):
  - `data/embeddings/ebola_metadata.csv` (rows: `1610`)

### Output folders (auto-created)
- `data/results/ebola_step04A/`
- `data/results/ebola_step04A/figures/`
- `data/interim/` (for combined FASTA and alignment outputs)

---

## 1) Environment warnings (IMPORTANT)
If you see this warning early:

> "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x…"

This means your environment may be inconsistent (common on local conda setups).  
It can break `.numpy()` conversions and sometimes SciPy/sklearn.

**Recommended fix (stable):**
- Downgrade numpy in that environment:
  - `pip install "numpy<2"`
  - or `conda install numpy<2 -y`
- Restart kernel.

**If you already completed the key steps without crashing, you can proceed**, but keep an eye on any code that converts Torch tensors to numpy arrays.

---

## SECTION 0 — Setup paths + imports
### Cell 0.1 — Imports + path configuration
**What it does**
- Imports libraries (BioPython, Torch, Pandas, etc.)
- Defines base directory and all important file paths:
  - reference FASTA
  - dataset FASTA
  - embeddings + metadata
  - output directories
  - interim alignment files

**Why it’s needed**
- Ensures every later cell can use the same consistent paths.
- Prints `exists()` checks to confirm your files are correctly located.

**Expected output**
You should see:
- `REF exists: True`
- `DATA exists: True`
- `EMB exists: True`
- `META exists: True`

If any is False: stop and fix paths before continuing.

---

## SECTION 1 — Load and validate (Reference + FASTA + metadata + embeddings)
### Cell 1.1 — Load reference (single FASTA)
**What it does**
- Reads reference file and asserts it contains exactly **1** sequence.
- Saves:
  - `ref_id`
  - `ref_seq`

**Why it’s needed**
- Mutation scoring requires a reference coordinate system:
  - ref position 1..N
- Later, we map aligned columns → reference positions.

---

### Cell 1.2 — Load dataset FASTA (Ebola cleaned)
**What it does**
- Loads Ebola sequences from `data/processed/ebola_cleaned.fasta`
- Builds `dataset_df` with columns:
  - `id`
  - `sequence`
  - `length`

**Why it’s needed**
- We need the dataset sequences to:
  - compare IDs with metadata,
  - create alignment including the reference,
  - compute per-site AA counts from the alignment.

---

### Cell 1.3 — Load metadata + embeddings and verify they match FASTA
**What it does**
- Loads:
  - `meta = pd.read_csv(META_CSV)`
  - `emb = torch.load(EMB_PT)`
- Verifies:
  - number of metadata rows == number of embedding rows
  - `id` sets match between FASTA and metadata
- Reorders metadata to match the FASTA ordering by ID
- Ensures `embedding_idx` is 0..N-1 after ordering (strict alignment check)
- Defines:
  - `X = emb.float()`

**Why it’s needed**
This prevents a very common fatal mistake:
> embeddings rows do not correspond to the same sequence IDs you think they do.

If this cell passes, your embeddings and metadata are properly aligned to your dataset.

---

## SECTION 2 — Alignment with reference (MAFFT)
### Cell 2.1 — Check MAFFT exists
**What it does**
- Runs `mafft --version` to confirm MAFFT is installed.

**Why it’s needed**
- We need a multiple sequence alignment so that:
  - every sequence site can be mapped to a **reference position**.

---

### Cell 2.2 — Write combined FASTA (reference first, then dataset)
**What it does**
- Builds a combined FASTA containing:
  1) the reference sequence first
  2) then all dataset sequences
- Writes:
  - `ebola_ref_plus_dataset.fasta`

**Why it’s needed**
- MAFFT aligns multiple sequences.
- Reference must be first so later code can assume aligned[0] is reference.

---

### Cell 2.3 — Run MAFFT to create MSA
**What it does**
- Runs:
  - `mafft --auto combined.fasta > aligned.fasta`
- Writes:
  - `ebola_with_reference_aligned.fasta`

**Why it’s needed**
- Provides the alignment required for:
  - per-site conservation,
  - entropy,
  - mapping alignment columns to reference positions.

---

## SECTION 3 — Build reference coordinate mapping
### Cell 3.1 — Load alignment and build aln_to_refpos mapping
**What it does**
- Loads aligned sequences and checks they all have the same alignment length.
- Builds:
  - `aln_to_refpos`: list length = alignment columns
    - contains reference position (1-based) where ref has an amino acid
    - contains `None` where reference has a gap

**Why it’s needed**
- Converts “alignment column i” into “reference position p”
- This makes all downstream mutation scoring interpretable.

---

## SECTION 4 — Site-level scores (conservation, entropy, category)
### Cell 4.1 — Compute per-reference-position counts
**What it does**
For each aligned column that corresponds to a real reference site:
- Counts amino acids across the dataset
- Computes:
  - `gap_rate`
  - `consensus_aa`
  - `conservation` (consensus frequency among non-gap AAs)
  - `entropy` (Shannon entropy)
  - `top_aas` (top 5 AAs and counts)

Produces:
- `site_df` with one row per reference position (Ebola ref length = 676)

**Why it’s needed**
This answers:
- Which sites are extremely conserved? (likely critical)
- Which sites are variable? (possible hotspots)
- Where alignment has poor coverage (high gap rate)?

---

### Cell 4.2 — Label sites: Critical / Hotspot / etc.
**What it does**
- Defines thresholds and labels each site into categories:
  - MostlyGap
  - Critical (very high conservation, low gaps)
  - Conserved
  - Hotspot
  - Intermediate

**Why it’s needed**
Gives you a simple interpretable classification of sites.

**Note:** In your output you saw many "Critical" sites—this might mean the threshold is too strict or entropy threshold logic needs tuning depending on alignment characteristics. Keep it, but review summary plots later.

---

### Cell 4.3 — Save site-level table
**What it does**
- Saves:
  - `data/results/ebola_step04A/Ebolaa_site_scores_reference_based.csv`

**Why it’s needed**
This is a main artifact you will compare later against Lassa.

---

## SECTION 5 — Substitution-level scoring (BLOSUM62)
### Cell 5.1 — BLOSUM62 and scoring functions
**What it does**
- Defines BLOSUM62 substitution matrix.
- Demonstrates example scores (e.g., A→V).
- Introduces a mutation impact scoring approach based on BLOSUM and other signals.

**Why it’s needed**
This supports the question:
> At a given reference position, is AA change X→Y likely harmless or disruptive?

This is a classical protein scoring baseline.

---

## Outputs produced by this notebook
You should end up with:

### Alignment / interim
- `data/interim/ebola_ref_plus_dataset.fasta`
- `data/interim/ebola_with_reference_aligned.fasta`

### Main results (Ebola Step 04)
- `data/results/ebola_step04A/Ebolaa_site_scores_reference_based.csv`
- (and later: substitution-level scoring tables, plots, etc.)

---

## What to run next
### If you are doing comparative analysis:
1) Run the equivalent Step 04 notebook for **Lassa**:
   - produce `Lassa_site_scores_reference_based.csv` (or same structure)
2) Then do Step 05:
   - compare:
     - site conservation distributions
     - entropy profiles
     - hotspot density
     - “critical site” overlap or differences between viruses

---

## Common failure points and fixes
### 1) FASTA vs metadata vs embeddings mismatch
- This notebook explicitly checks ID match and order.
- If mismatch occurs:
  - rebuild metadata from the same FASTA you embedded
  - ensure you didn’t shuffle sequences between embedding step and analysis step

### 2) MAFFT not installed
- Install MAFFT (conda or brew), restart kernel.

### 3) NumPy 2.x compiled module warning
- Best fix: `numpy<2`, restart kernel.

---

## Notes on sequence lengths
Your Ebola sequences appear very long in the dataset (lengths ~5,700+), while the reference is length 676. This is okay only if your FASTA is indeed protein sequences for the same target and alignment is meaningful.

If those are nucleotide or concatenated polyprotein sequences, the alignment approach must be revisited (protein vs nucleotide alignment). Double-check what the sequences represent.

---
**Last updated:** 2026-03-31