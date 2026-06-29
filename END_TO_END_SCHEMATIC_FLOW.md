# End-to-End Schematic Flow: Comparative Lassa–Ebola Sequence Analysis & Prediction System

## 1. Executive Summary

This document captures the complete, reproducible computational pipeline for **comparative mutational analysis and computational atypicality scoring of Lassa virus (LASV) and Ebola virus (EBOV) glycoprotein sequences**. The system integrates raw sequence acquisition, cleaning, protein-language-model embedding, entropy/site-category analysis, a lightweight composition-based classifier, atypicality scoring, and an interactive Streamlit web application with PDF export.

### Novelty of the current flow
- **Composition-only classifier** trained on 21 low-cost features (length + amino-acid frequencies) achieves **perfect separation** of LASV vs EBOV GP sequences, indicating that the two viruses are compositionally distinct at the GP level.
- **ESM-2 embedding layer** is used for *atypicality monitoring*, not for the core label prediction; it flags statistically rare variants that may warrant surveillance attention.
- **Risk calibration** is an explicit *distance-from-centroid* computation, clearly separated from validated clinical risk, and labeled as **computational atypicality** rather than biological pathogenicity.
- **Deployment-ready QA** is built into the repository: reference panels, edge-case panels, grouped validation, ablation studies, and PDF/CSV export.

---

## 2. High-Level Flow Diagram (Text)

```
Raw FASTA (NCBI / EpiFlu / manuscript downloads)
        │
        ▼
┌────────────────────┐
│ 1. Data Acquisition │   │ Outputs: ebola_raw.fasta, lassa_raw.fasta
└────────────────────┘
        │
        ▼
┌────────────────────┐
│ 2. Sequence Cleaning│   │ Remove ambiguity, strip non-canonical AAs, drop short fragments
└────────────────────┘
        │
        ▼
┌────────────────────┐
│ 3. Metadata Parsing │   │ Parse accession, country, date, lineage from FASTA IDs
└────────────────────┘
        │
        ▼
┌────────────────────────────────────────┐
│ 4. Feature Engineering                  │
│   ├── Composition features (length + 20 AA frequencies)  → used by classifier
│   └── ESM-2 embeddings (esm2_t12_35M_UR50D)          → used for atypicality/outlier analysis
└────────────────────────────────────────┘
        │
        ▼
┌────────────────────┐
│ 5. Model Training     │   │ Logistic Regression + StandardScaler vs Random Forest
│   └── Calibration    │   │ Class centroids, distance stats, risk bands
└────────────────────┘
        │
        ▼
┌────────────────────┐
│ 6. Validation & QA  │   │ Stratified CV, grouped CV, ablation, reference/edge-case panels
└────────────────────┘
        │
        ▼
┌────────────────────┐
│ 7. Site-Level Analysis│   │ Entropy, conservation categories, site categories for GP alignment
└────────────────────┘
        │
        ▼
┌────────────────────┐
│ 8. Streamlit App    │   │ Upload FASTA/CSV/TXT → classify → atypicality score → charts → export
└────────────────────┘
```

---

## 3. Data Flow Detail

### 3.1 Data Acquisition & Composition

| Dataset | File | Count | Length Profile |
|---|---|---|---|
| LASV GP | `data/cleaned/cleaned_sequences_gp_only.csv` | 780 | mean 1,057 aa (std 55) |
| EBOV Makona GP | `data/cleaned/cleaned_sequences_gp_only.csv` | 1,722 | mean 669 aa (std 30) |
| Full cleaned set (including other segments) | `data/cleaned/cleaned_sequences.csv` | 2,390 | 1610 EBOV, 780 LASV |

- **Duplicate handling**: 185 exact duplicate sequences flagged across the full 2,390-sequence set.
- **Metadata schema**: `sequence_id`, `virus`, `accession_id`, `length`, `duplicate_flag`.
- **Reference panels**: `data/Reference/Lassa_Reference_Sequence.fasta`, `data/Reference/Ebola_Reference_Sequence.fasta`.

### 3.2 Sequence Cleaning

Implemented in `src/features/sequence_features.py`:

```python
def clean_sequence(sequence: str) -> str:
    sequence = sequence.upper()
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", sequence)
```

- All non-canonical amino acids (`B`, `J`, `X`, `Z`, gaps, numeric characters, whitespace) are stripped.
- The classifier therefore only sees the 20 standard residues.

### 3.3 Feature Engineering

**Composition features (`amino_acid_frequency_features`)**:
- `seq_length`: length of cleaned sequence
- `aa_freq_A` … `aa_freq_Y`: per-residue frequency of each of the 20 canonical amino acids

**ESM-2 embeddings**:
- Model: `esm2_t12_35M_UR50D` (12 layers, 35M params, 480-dim per-token embeddings)
- Mean-pooled per-token representations for each sequence
- Stored in `notebooks/embedding/all_embeddings_COMPLETE.pt` with metadata CSV

### 3.4 Training & Calibration

**Script**: `scripts/03_train.py` → calls `src/models/train.py`.

**Candidate models**:
- `logistic_regression`: Pipeline(StandardScaler + LogisticRegression(max_iter=300, random_state=42))
- `random_forest`: 300 trees, balanced class weights, random_state=42

**Selection**: Best by F1 on 80/20 stratified holdout. Both models achieve F1 = 1.000 on the current dataset.

**Risk calibration** (`_build_risk_calibration`):
1. Fit `StandardScaler` on `X_train`.
2. Compute scaled class centroids for LASV (0) and EBOV (1).
3. For each prediction, compute Euclidean distance from the predicted-class centroid.
4. Convert distance to z-score using class-specific mean/std.
5. Map z-score to `atypicality_index = clamp(50 + 15 × z, 0, 100)`.
6. Band: Low (<20), Below-Average (<40), Average (<60), Elevated (<80), High (≥80).

**Model artifact**: `models/gp_revision/gp_classifier_v2.joblib` contains:
- `model`: fitted sklearn model/pipeline
- `risk_calibration`: centroids, distance stats, scaler mean/scale, feature columns
- `feature_columns`: ordered column names for inference alignment

### 3.5 Inference & Atypicality Scoring

**Script**: `src/models/predict.py` → `predict_sequences()`.

1. Load model bundle once (cached in `_BUNDLE_CACHE`).
2. Clean each input sequence.
3. Build composition features and align columns to training order.
4. Call `model.predict_proba()` — the Pipeline handles scaling internally.
5. Compute predicted class and confidence.
6. Compute atypicality score from calibrated class centroid distance.
7. If `atypicality_index ≥ 95` or `z-score ≥ 3.0`, label as **Unknown / Highly Atypical** instead of a binary virus label.

---

## 4. Validation, QA & Ablation

### 4.1 Validation Strategy

Run by `scripts/rnr_revision_analysis.py`.

| Validation | Description | Result (Logistic Regression) |
|---|---|---|
| Repeated Stratified CV | 5-fold × 3 repeats | accuracy = 1.000, F1 = 1.000 |
| Grouped CV (GroupKFold) | Groups from lineage/country/year | accuracy = 1.000, F1 = 1.000 |
| External Group Holdout | 80/20 grouped split | accuracy = 1.000, F1 = 1.000 |

### 4.2 Ablation Study

| Feature Set | Logistic Regression F1 | Interpretation |
|---|---|---|
| Composition only | 1.000 | Sufficient for virus label task |
| ESM-2 only | 1.000 | Also sufficient |
| Composition + ESM-2 | 1.000 | No gain on this binary task |

### 4.3 QA Panels

| Panel | Inputs | Acceptance Criteria |
|---|---|---|
| Reference | Canonical Lassa & Ebola reference sequences | Correct labels returned, all fields present |
| Edge-case | `ACD`, `AXXZ---TT??`, `A12345VVVV`, long repeat | App does not crash; returns prediction + warning |

Outputs archived in `validation_artifacts/tables/qa_reference_predictions.csv` and `qa_edge_case_predictions.csv`.

---

## 5. Site-Level / Biological Analysis

### 5.1 Entropy & Site Categories

- Alignment-derived entropy computed per GP position for LASV and EBOV separately.
- Categories assigned from entropy thresholds: `Critical`, `Conserved`, `Hotspot`, `Intermediate`, etc.
- Files: `results/gp_revision/site_categories.csv`, `site_categories_aligned.csv`.

### 5.2 ESM-2 Outlier Analysis

- 344 EBOV sequences (20.0%) flagged as >80th percentile outliers in ESM-2 space.
- 156 LASV sequences (20.0%) flagged as outliers.
- High-atypical tail: 38 EBOV, 31 LASV.
- PCA of embeddings: PC1 explains 57.8% variance; PC2 explains 15.8%.

### 5.3 Key Biological Finding

LASV and EBOV GP sequences are compositionally distinct enough that a simple 21-feature logistic regression separates them perfectly. The ESM-2 layer adds value not by improving label accuracy but by **identifying rare, embedded-space variants that may be candidates for further surveillance**.

---

## 6. Deployment: Streamlit App

**File**: `app.py`
**URL**: `https://mutation-analysis.streamlit.app`

### 6.1 App Capabilities

1. **Upload**: FASTA, CSV (with `sequence` column), or plain text (one sequence per line).
2. **Clean & predict**: Composition features → classifier → confidence + atypicality score.
3. **Visualise**: Class distribution, composition deviation, atypicality bands, composition bar chart.
4. **Report cards**: Per-sequence summary with risk band and disclaimer.
5. **Export**: CSV + landscape PDF report via `reportlab`.

### 6.2 Deployment Constraints

- Python pinned to 3.10 via `.python-version`.
- Heavy `torch` / `fair-esm` dependencies **removed** from `requirements.txt` because the deployed app uses composition-only classification; ESM is an offline analysis layer.
- Model is retrained with the scikit-learn version available in the deployment environment to avoid version mismatch errors.

### 6.3 Disclaimer

The app explicitly states that atypicality scores are **statistical deviation indices**, not validated clinical risk assessments, and should not be used for diagnostic or therapeutic decisions without independent experimental validation.

---

## 7. File Reference Map

| Purpose | Path |
|---|---|
| Main app | `app.py` |
| Model training | `scripts/03_train.py`, `src/models/train.py` |
| Inference | `src/models/predict.py` |
| Feature engineering | `src/features/sequence_features.py` |
| Configuration | `src/config.py` |
| Revision analysis | `scripts/rnr_revision_analysis.py` |
| Trained model | `models/gp_revision/gp_classifier_v2.joblib` |
| Training metrics | `models/gp_revision/training_metrics.json` |
| Dataset | `data/cleaned/cleaned_sequences_gp_only.csv` |
| Embeddings | `notebooks/embedding/all_embeddings_COMPLETE.pt` |
| Validation tables | `validation_artifacts/tables/` |
| Validation figures | `validation_artifacts/figures/` |
| Validation docs | `validation_artifacts/docs/` |
| Manuscript | `manuscript/Comparative_Analysis_Refined_3_1.md` |
| Reviewer response | `manuscript/reviewer_3_response.md` |
| Flow diagram | `figures/flowchart_vertical_final.py` → `figures/flowchart.png` |

---

## 8. Reproducibility Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the classifier
cd /Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model
python scripts/03_train.py

# 3. Run validation & QA (requires torch + fair-esm offline)
python scripts/rnr_revision_analysis.py

# 4. Run local Streamlit app
streamlit run app.py

# 5. Quick inference check
python -c "
from src.models.predict import predict_sequences
seq = ['MGLYKIWLVLFLVALVAATGVPSLSEGIVNDVRQLMPHSLNSTQSLDQSVQTLDLNRTGKLLQLTDQRIQALNEAESLTEAQAQAEVQQQAEAALQQA']
print(predict_sequences(seq))
"
```

---

## 9. Intellectual Property / Patent Notes

What is new and potentially protectable in this end-to-end system:

1. **Integrated dual-layer architecture** where a lightweight composition classifier handles the virus-label task and a protein-language-model embedding layer provides an *independent* atypicality signal.
2. **Calibration-based atypicality index** that maps a sequence’s distance from a class-specific centroid in scaled composition space onto a 0–100 scale with explicit bands.
3. **Abstention rule** (`Unknown / Highly Atypical`) triggered by high atypicality index or z-score, preventing over-confident classification of out-of-distribution sequences.
4. **Deployment-specific QA pipeline** combining reference panels, edge-case panels, grouped validation, and ablation studies, designed for a regulated/public-health deployment context.
5. **Public-facing web application** with transparent disclaimers and PDF reporting, translating a computational model into a reusable surveillance tool.

Recommended next steps for patent/IP documentation:
- Finalise the exact set of claims (method claim, system claim, computer-readable medium claim).
- Document all hyperparameters and thresholds in the calibration step.
- Prepare a working example (e.g. 3–5 real sequences) that demonstrates the outlier flagging behavior.
- Draft a provisional patent application or technical memorandum with inventorship confirmed.

---

## 10. Glossary

| Term | Definition |
|---|---|
| **Composition features** | Length + frequency of each of the 20 canonical amino acids. |
| **ESM-2** | Meta AI protein language model used here to generate sequence embeddings. |
| **Atypicality index** | 0–100 score derived from distance to the predicted class centroid in scaled composition space. |
| **Atypicality z-score** | Standardised deviation from the mean class centroid distance. |
| **Grouped CV** | Cross-validation where all sequences from the same lineage/country/year are kept together. |
| **Ablation** | Comparison of model performance using different feature subsets. |

---

*Document generated for the project:* `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model`  
*GitHub:* `https://github.com/Damilola-max/Comparative_Lassa_Ebola-Model.git`  
*Streamlit app:* `https://mutation-analysis.streamlit.app`
