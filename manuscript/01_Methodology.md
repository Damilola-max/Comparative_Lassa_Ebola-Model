# 2.0 Methodology

## 2.1 Study Design
This work was designed as an end-to-end computational virology pipeline with two linked objectives:  
(i) characterize and compare mutation landscapes in Lassa and Ebola proteins; and  
(ii) deploy an interpretable sequence-level prediction tool for practical use.

The workflow integrates data curation, site-level and substitution-level comparative analysis, embedding-space analysis, supervised classification, and a public Streamlit deployment.

## 2.2 Data Sources and Provenance
Sequence collection was anchored to Ebola and Lassa FASTA resources used in the project notebooks.

### Ebola sources
1. Makona genomes FASTA (project-specified source):  
   `https://github.com/ebov/space-time/blob/master/Data/Makona_1610_genomes_2016-06-23.fasta`
2. Nextstrain Ebola context (project-specified source):  
   `https://nextstrain.org/ebola/ebov-2013`
3. Repository raw Ebola FASTA used in pipeline download steps:  
   `https://raw.githubusercontent.com/Damilola-max/Comparative_Lassa_Ebola-Model/main/data/raw/Ebola_Protein_Sequence.fas`

### Lassa source
Repository raw Lassa FASTA used in pipeline download steps:  
`https://raw.githubusercontent.com/Damilola-max/Comparative_Lassa_Ebola-Model/main/data/raw/S_protein.fas`

### Final analysis cohort
After cleaning and validation, the unified dataset (`data/cleaned/cleaned_sequences.csv`) contained 2,390 sequences (780 Lassa, 1,610 Ebola), with metadata in `data/cleaned/sequence_metadata.csv` and cohort summary in `data/cleaned/preprocessing_summary.json`.

## 2.3 Sequence Cleaning and Standardization
All sequences were converted to uppercase and filtered to canonical amino-acid symbols (`ACDEFGHIKLMNPQRSTVWY`) using deterministic cleaning logic (`src/features/sequence_features.py`). Non-canonical symbols (e.g., `?`, `*`, `-`) were removed prior to feature extraction.

## 2.4 Comparative Mutation-Landscape Pipeline
Comparative analysis outputs were produced in the 05C stage and stored in:
- `results/05C_Result/05C_table/site_category_comparison.csv`
- `results/05C_Result/05C_table/conservation_entropy_summary_stats.csv`
- `results/05C_Result/05C_table/observed_substitution_comparison.csv`
- `results/05C_Result/05C_table/embedding_comparison_stats.csv`
- `results/05C_Result/05C_table/outlier_score_comparison.csv`

The analysis framework includes:
- site-category profiling (Critical, Conserved, Hotspot, Intermediate, MostlyGap),
- conservation and entropy distribution comparisons,
- observed versus unobserved substitution burden,
- embedding centroid-distance and separation summaries,
- outlier-score distribution analysis.

## 2.5 Predictive Modeling
Two supervised baselines were trained for Lassa-vs-Ebola classification (`src/models/train.py`):
1. Logistic Regression (with scaling),
2. Random Forest.

A stratified 80/20 split was used (`RANDOM_STATE=42`). Selection was based on F1 score, with metrics persisted in `models/final/training_metrics.json`.

## 2.6 Inference and Interpretability Layer
The deployed inference module (`src/models/predict.py`) returns:
- predicted class,
- confidence and Ebola probability,
- mutation risk score (0–100),
- risk category (Harmless/Neutral/Moderate/Dangerous/Critical),
- atypicality z-score.

Risk scoring is distance-based in standardized feature space relative to predicted-class centroids, supporting transparent triage-oriented interpretation.

## 2.7 Deployment and Reproducibility
Public deployment: `https://mutation-analysis.streamlit.app`  
Application entrypoint: `app.py`.

Reproducible local run:
```bash
pip install -r requirements.txt
python3 scripts/03_train.py
python3 scripts/04_evaluate.py
streamlit run app.py
```

## 2.8 Suggested Figure Placement (Methods)
- **Figure 1**: Pipeline overview (data provenance → cleaning → comparative analysis → modeling → app deployment).
- **Figure 2**: Inference architecture and risk derivation.
- **Figure 3**: Deployed interface/report-card snapshot.

## 2.9 Methodological Scope
This framework is designed for computational virology research and sequence triage support. Risk outputs are model-derived interpretive indicators and should not be treated as clinical diagnostic probabilities without independent biological and clinical validation.
