# Supplementary Methods and Validation Report (R&R)

## Scope
This report documents additional analyses and validation work performed to address reviewer concerns on dataset transparency, model validation rigor, feature attribution (composition vs ESM), and interpretation limits.

## Reproducibility Context
- Project root: `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model`
- Main analysis script added for R&R: `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/scripts/rnr_revision_analysis.py`
- Output folder: `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision`

## 1) Dataset Transparency Expansion
A dataset manifest and summary tables were generated:
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/tables/dataset_manifest.csv`
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/tables/dataset_virus_summary.csv`
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/tables/dataset_missingness.csv`

### Key observations
- Total sequences: 2,390
- LASV: 780
- EBOV: 1,610
- Exact duplicate sequences globally: 107 (EBOV: 101, LASV: 6)
  - Source: `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/tables/duplicate_summary.csv`

### Important clarification
Metadata fields for LASV were parsed from sequence IDs where available and filled with `unknown` when absent. This is why missingness by `NaN` is low while `unknown` rates remain non-trivial (e.g., ~32.6% for `country`, `collection_date`, and `host`). This limitation should be stated in the manuscript.

## 2) Leakage Controls and Validation Design
### Added validation strategies
- Repeated stratified CV (5 folds × 3 repeats) on composition features.
- Grouped CV (`GroupKFold`) using source-aware grouping keys.
- External grouped holdout using `GroupShuffleSplit`.

### Output files
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/tables/validation_summary.csv`
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/tables/external_group_holdout_metrics.csv`

### Key finding
Even under grouped validation, the class task remains highly separable (near-perfect metrics). This strongly suggests the classification task is dominated by broad sequence-level separability between LASV and EBOV, not subtle mutation-risk inference.

## 3) ESM Ablation Study
Ablation performed with three feature sets:
1. Composition-only
2. ESM-only
3. Composition + ESM

### Output files
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/tables/ablation_summary.csv`
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/figures/ablation_f1_logistic.png`

### Key finding
All three feature sets performed similarly (near-perfect), with composition-only already sufficient for the binary virus-label task. Therefore, claims that ESM is the central driver of classifier performance should be moderated unless a harder task and stronger external validation are introduced.

## 4) Risk Scoring Interpretation Check
Current deployed scoring maps atypicality distance into a `0–100` scale and labels it using clinical-sounding bins (`Harmless` to `Critical`). Additional QA indicates this should be interpreted as model atypicality rather than biological mutation risk.

Reference outputs:
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/tables/qa_reference_predictions.csv`
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/tables/qa_edge_case_predictions.csv`

## 5) Figures Produced for Revision Package
- Validation split comparison: `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/figures/validation_split_comparison.png`
- Ablation performance comparison: `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/figures/ablation_f1_logistic.png`
- QA reference panel snapshot: `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/figures/qa_reference_predictions.png`
- QA edge-case panel snapshot: `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/figures/qa_edge_case_predictions.png`

## 6) Manuscript Language Changes Recommended
- Replace `operational risk stratification` with `computational atypicality scoring`.
- Replace `mutation risk score` with `atypicality index` in text and figure labels where biologic validation is not provided.
- Remove certainty phrases (e.g., `single, unambiguous conclusion`) and add explicit limitations around dataset representativeness and label separability.

## 7) Remaining Scientific Gap Before Resubmission
To defend stronger biological interpretation, add at least one biological validation layer:
- map high-scoring variants to known functional domains/epitopes,
- compare against known escape/fitness mutations,
- or test whether model flags curated biologically meaningful variants.

Without this layer, the ML component should be framed as computational classification and atypicality monitoring, not validated mutation-risk prediction.
