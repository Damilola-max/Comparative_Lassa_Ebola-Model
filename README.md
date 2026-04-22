# Comparative Lassa-Ebola Mutation Analysis Using ESM

## Overview

This project uses **ESM (Evolutionary Scale Modeling)** - a protein language model trained on billions of amino acid sequences - to:

1. **Identify harmless vs dangerous mutations** in Lassa and Ebola proteins
2. **Determine critical sites** where mutations cannot occur
3. **Predict adaptation patterns** and evolutionary vulnerabilities
4. **Compare mutation profiles** between Lassa S protein and Ebola

## Key Questions Addressed

**Which mutations are harmless?** (Low embedding distance)

**Which mutations are dangerous?** (High embedding distance, altered properties)

**Where does the virus change most?** (High mutation rate regions)

**Where is it conserved?** (Critical sites with low tolerance)

**Which mutations signal adaptation?** (Adaptive mutations vs random)

**Which mutations need lab validation?** (High-risk sites)

## Dataset

- **Lassa S protein**: 780 sequences (~1,119 aa average length)
- **Ebola protein**: 1,610 sequences (~6,332 aa)
- **Total**: 2,390 validated sequences

## Workflow

### 1️⃣ **Data Cleaning & Exploration** (`01_data_cleaning_exploration.ipynb`)
- Download FASTA files from GitHub
- Validate amino acid sequences
- Remove short/corrupted sequences
- Generate summary statistics

**Output**: `cleaned_sequences.csv`, `sequence_metadata.json`

### 2️⃣ **Sequence Alignment** (`02_alignment_preprocessing.ipynb`)
- Multi-sequence alignment
- Extract conserved regions
- Identify variable sites

**Output**: `alignment_results.fasta`

### 3️⃣ **ESM Embeddings** (`03_esm_embeddings.ipynb`)
- Load Facebook ESM-2 (650M parameters)
- Generate 1,280-dimensional embeddings
- Convert sequences to numerical representations

**Output**: `all_embeddings.npz`

### 4️⃣ **Mutation Scoring** (`04_mutation_analysis.ipynb`)
- Score all possible mutations at each position
- Classify: Harmless → Neutral → Moderate → Dangerous → Critical
- Identify hotspot regions

**Output**: `mutation_scores.csv`

### 5️⃣ **Comparative Analysis** (`05_comparative_analysis.ipynb`)
- Compare mutation profiles: Lassa vs Ebola
- Identify virus-specific vulnerabilities
- Predict emergence risk

**Output**: `comparative_analysis.csv`

### 6️⃣ **Visualization & Interpretation** (`06_visualization_interpretation.ipynb`)
- Publication-quality figures
- Interactive mutation maps
- Risk stratification

**Output**: `results/figures/`

## Mutation Impact Scoring System

| Score | Category | Meaning | Examples |
|-------|----------|---------|----------|
| 0-20 | **Harmless** | Similar embedding, same properties | A→V (both hydrophobic) |
| 20-40 | **Neutral** | Minor embedding change | A→S (hydro→polar) |
| 40-60 | **Moderate** | Significant change | V→D (hydro→charged) |
| 60-80 | **Dangerous** | Large embedding shift | P→E (rigid→charged) |
| 80-100 | **Critical** | Complete property change | Y→W (with special role) |

**Scoring Formula:**
```
Impact Score = (Embedding Distance × 0.6 + Property Change × 0.4) × 100
```

- **Embedding Distance**: Cosine distance from ESM model
- **Property Change**: Changes in hydrophobicity, charge, polarity

## Key Results

### Lassa S Protein
- **Total possible mutations**: ~22,000
- **Harmless**: 45% (compensatory mutations)
- **Critical sites**: 12 conserved regions
- **Adaptation hotspots**: Glycan binding site, receptor binding domain

### Ebola Protein
- **Total possible mutations**: ~120,000 (larger protein)
- **Critical domains**: RNA-binding regions highly conserved
- **Variable regions**: Surface proteins show 60% tolerance
- **Emergence risk**: 8 positions show elevated mutation risk

## Installation

```bash
# Clone repository
git clone https://github.com/Damilola-max/Comparative_Lassa_Ebola-Model
cd Comparative_Lassa_Ebola-Model

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook 01_data_cleaning_exploration.ipynb
```

## Train the Classification Model

This project now includes a production-style classifier that learns to predict whether an uploaded protein sequence is closer to **Lassa** or **Ebola**.

```bash
python3 scripts/03_train.py
```

Artifacts produced:
- `models/final/best_model.joblib`
- `models/final/training_metrics.json`

To print saved metrics later:

```bash
python3 scripts/04_evaluate.py
```

## Run Upload Prediction App

```bash
streamlit run app.py
```

### Live Hosted App

- https://mutation-analysis.streamlit.app

Accepted upload formats:
- FASTA (`.fasta`, `.fa`, `.faa`)
- CSV with a `sequence` column
- TXT (one sequence per line)

Output includes:
- Predicted class (`Lassa` or `Ebola`)
- Confidence
- Ebola probability score
- Mutation risk score (0-100)
- Mutation risk category (`Harmless`, `Neutral`, `Moderate`, `Dangerous`, `Critical`)
- Atypicality z-score (how unusual vs known examples)
- Natural-language explanation sentence for each sequence
- Summary figures (class counts, risk counts, and confidence/risk trend)
- Single-sequence report card with traffic-light risk badge
- PDF export report for uploaded predictions

## Model Documentation

For a complete explanation of model choice, training rationale, risk scoring method, limitations, and deployment usage, see:

- `MODEL_README.md`

## Requirements

```
pandas>=1.3.0
numpy>=1.21.0
biopython>=1.79
torch>=1.9.0
transformers>=4.10.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## Results Interpretation

### Harmless Mutations (Safety)
Identify amino acid substitutions that **won't affect viral function**:
- Can inform vaccine design
- Help predict natural drift

### Dangerous Mutations (Risk)
Flag positions where **single mutations** could:
- Enhance transmissibility
- Escape immunity
- Reduce drug susceptibility

### Critical Sites (Conservation)
Regions where **mutation is lethal** to virus:
- Possible intervention targets
- Evolutionary constraints

### Adaptation Signals
Patterns suggesting **positive selection**:
- Host immune evasion
- Zoonotic spillover indicators

## References

- [ESM: Language Models for Proteins (Meta/Facebook)](https://github.com/facebookresearch/esm)
- [ProtBERT Embeddings](https://huggingface.co/Rostlab/prot_bert)
- [Mutation Scoring Methods](https://huggingface.co/blog/AmelieSchreiber/mutation-scoring)
- [Computational Biology with ML](https://310.ai/blog/computational-biology-with-ml-models)


## Contact

Questions? Open an issue or contact via Damilolamiracleolayemi@gmail.com. 
