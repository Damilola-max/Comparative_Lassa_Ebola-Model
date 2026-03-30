# Step 04A — Lassa Hybrid Mutation Scoring (Reference + Alignment + ESM)

## What this step is (in plain words)

This step builds a **Lassa mutation-risk scoring system** using **three inputs**:

1. **A Reference Sequence (FASTA)**  
   Used to define stable position numbering (e.g., “mutation at position 123”).

2. **A Dataset of Lassa sequences (FASTA)**  
   Used to measure how much each site varies across real sequences (conserved vs variable).

3. **ESM-2 embeddings for the dataset (Tensor + CSV)**  
   Used to detect **unusual/outlier sequences** and hidden clusters in embedding space.

This is a **hybrid approach**:
- **Alignment-based scoring** → answers “critical sites”, “hotspots”, “which substitutions are risky at a position”.
- **ESM embedding scoring** → answers “is this sequence typical or an outlier”, “nearest neighbors”, “possible adaptation signals”.

This step produces outputs that support the project’s key questions:
- Which mutations are harmless vs dangerous?
- Which sites are conserved/critical (low tolerance)?
- Where are hotspots (high variation)?
- Which sequences look most unusual/adaptive?
- Which sites/substitutions should be prioritized for lab validation?

---

## Inputs (what files you need)

The notebook expects these Lassa files:

### 1) Reference sequence
- **`/Users/user/data/Reference/Lassa_Reference_Sequence.fasta`**
  - Must contain **exactly 1 protein sequence**
  - This defines `ref_pos` numbering

### 2) Lassa cleaned dataset
- **`/Users/user/data/processed/lassa_cleaned.fasta`**
  - Contains the cleaned Lassa sequences (e.g., ~780 sequences)

### 3) ESM embeddings + metadata mapping
- **`/Users/user/data/embeddings/lassa_embeddings.pt`**  
  - A `torch.Tensor` of shape `(N, 1280)` where `N = number of sequences`
- **`/Users/user/data/embeddings/lassa_metadata.csv`**
  - Must contain at minimum: `id`, `sequence`, `embedding_idx`, `length`
  - IDs must match the FASTA IDs

---

## Outputs (what this step produces)

All outputs are written to:

- **`/Users/user/data/results/lassa_step04A/`**

### A) Alignment outputs (intermediate but important)
- `processed/lassa_ref_plus_dataset.fasta`  
  Combined FASTA (reference first, then dataset sequences)

- `processed/lassa_with_reference_aligned.fasta`  
  Multiple Sequence Alignment (MSA) output from MAFFT

### B) Site-level scores (per reference position)
- **`lassa_site_scores_reference_based.csv`**

Each row corresponds to one **reference position** (`ref_pos`) and contains:

- `ref_pos` — 1-based position in the reference sequence  
- `ref_aa` — amino acid in the reference at that position  
- `consensus_aa` — most common AA observed in dataset at that aligned position  
- `gap_rate` — fraction of dataset sequences that have a gap at this aligned column  
- `conservation` — fraction of dataset sequences (excluding gaps) that match the most common AA  
- `entropy` — Shannon entropy of AA distribution (higher = more variable site)  
- `site_category` — one of:
  - **Critical**: highly conserved (low tolerance)
  - **Conserved**: mostly conserved
  - **Hotspot**: highly variable (likely tolerant)
  - **Intermediate**
  - **MostlyGap**: alignment column mostly gaps

**Interpretation:**
- **Critical sites** are candidate “cannot mutate” sites (strong evolutionary constraint).
- **Hotspots** are positions where the virus changes most (high variation).

### C) Mutation substitution table (A→V vs A→D at each position)
- **`lassa_mutation_scores_reference_based.csv`**

This table is the “mutation scoring engine”.  
It lists a score for every substitution **reference AA → alt AA** for each reference position.

Columns:
- `ref_pos`, `ref_aa`, `alt_aa`
- `alt_freq_in_dataset` — how frequently `alt_aa` appears at that site in real sequences
- `blosum62` — substitution similarity score (biologically-informed)
- `site_conservation`, `site_entropy`, `site_category`
- `impact_score` (0–100) — final risk/impact score
- `impact_category` — Harmless / Neutral / Moderate / Dangerous / Critical

**How the impact score is computed:**
We combine:
1. **Site conservation** (more conserved site → higher risk)
2. **BLOSUM penalty** (biochemically unlikely substitutions → higher risk)
3. **Rarity penalty** (unobserved or rare substitutions → higher risk)

This is a dataset-driven prioritization score (not experimental truth).

### D) ESM outlier report (sequence-level novelty / adaptation proxy)
- **`lassa_esm_outlier_report.csv`**

Each row corresponds to a dataset sequence and contains:
- `cos_dist_to_centroid` — cosine distance to mean embedding (centroid)
- `esm_outlier_score` (0–100) — robust z-score mapped to a 0–100 scale
- `esm_outlier_category` — Harmless → Critical (same bins)

**Interpretation:**
- High outlier score = sequence is unusual compared to typical Lassa sequences.
- This can indicate:
  - potential adaptation
  - unusual lineage
  - sequencing artefact (always validate)

---

## Step-by-step: what the notebook actually does

### Step 1 — Load & validate
- Loads reference, dataset FASTA, metadata CSV, embedding tensor
- Validates:
  - IDs match between FASTA and metadata
  - embeddings shape matches metadata row count

### Step 2 — Align dataset sequences to the reference (MAFFT)
- Creates a combined FASTA: reference + dataset
- Runs MAFFT to produce an MSA

**Why this matters:**  
Without alignment, “position 123” is not comparable across sequences.

### Step 3 — Build reference coordinate mapping
- Converts alignment columns → reference positions (`ref_pos`)
- Any alignment column where the reference has a gap is excluded from reference numbering

### Step 4 — Compute site-level conservation + entropy
- For each `ref_pos`, counts AA frequencies across aligned dataset sequences
- Computes conservation and entropy
- Labels site_category = Critical / Hotspot / etc.

### Step 5 — Compute substitution-level mutation scores
For each position:
- For each alt amino acid (19 substitutions):
  - get frequency in dataset
  - get BLOSUM62 score
  - compute `impact_score` and category

This enables queries like:
> “At ref_pos 123, is A→D riskier than A→V?”

### Step 6 — ESM embedding outlier analysis
- Computes centroid of embeddings
- Computes cosine distance of each sequence to centroid
- Converts into robust outlier score (0–100)
- Computes nearest neighbors using cosine similarity

---

## How to answer key questions using the outputs

### “Which sites are critical (cannot mutate)?”
Filter `lassa_site_scores_reference_based.csv` where:
- `site_category == "Critical"`

### “Where are mutation hotspots?”
Filter site table where:
- `site_category == "Hotspot"` or entropy is high

### “Which substitutions are dangerous at a specific position?”
Filter `lassa_mutation_scores_reference_based.csv`:
- `ref_pos == X`
- sort by `impact_score` descending

### “Which sequences look adaptive/unusual?”
Sort `lassa_esm_outlier_report.csv` by `esm_outlier_score` descending.

---

## Important notes / limitations (read this)

### 1) Reference length vs dataset length matters
If your dataset sequences are ~1,000 aa but the reference is ~491 aa, then this pipeline scores only the portion covered by the reference.  
For full-length Lassa S/GPC scoring, use a **full-length reference** for that protein.

### 2) This is not experimental validation
The mutation scores are **prioritization** based on:
- population variation + BLOSUM similarity + rarity
They help flag high-risk mutations/sites but do not replace lab experiments.

### 3) Embedding outliers can include artifacts
High ESM outlier score may reflect:
- real biological novelty
- OR data quality issues (truncation, sequencing errors, unusual preprocessing)

---

## Next steps (what comes after Step 04A)

### Step 04B — Ebola
Repeat the same hybrid process for Ebola:
- Reference FASTA (Ebola protein reference)
- Ebola cleaned dataset FASTA
- Ebola embeddings + metadata  
Outputs:
- `ebola_site`
