# 3.0 Results

## 3.1 Analysis Cohort and Output Overview
After sequence cleaning and validation, the final analytical cohort included 2,390 protein sequences (780 Lassa; 1,610 Ebola). Comparative mutation-landscape outputs and summary tables were produced under `results/05C_Result/05C_table/`, while model-evaluation outputs were stored in `models/final/`.

For clarity, the core quantitative files used in this section are:
- `site_category_comparison.csv`
- `conservation_entropy_summary_stats.csv`
- `observed_substitution_comparison.csv`
- `embedding_comparison_stats.csv`
- `outlier_score_comparison.csv`
- `training_metrics.json`

## 3.2 Site-Level Mutational Regime Comparison
Site-category analysis demonstrated a highly asymmetric mutational regime between the two viruses.

### Ebola
- Total reference positions: 676
- Critical: 669 (98.9645%)
- Conserved: 6 (0.8876%)
- MostlyGap: 1 (0.1479%)
- Hotspot: 0 (0.0000%)
- Intermediate: 0 (0.0000%)

### Lassa
- Total reference positions: 491
- Critical: 2 (0.4073%)
- Conserved: 34 (6.9246%)
- Hotspot: 171 (34.8269%)
- Intermediate: 200 (40.7332%)
- MostlyGap: 84 (17.1079%)

The dominant Critical-site burden in Ebola indicates strong positional constraint in the analyzed reference context, whereas Lassa exhibits a wider positional tolerance profile through substantial Hotspot and Intermediate fractions.

**[Figure 4 here: Site-category comparison and barcode view]**  
Use comparative site-category plot and category-track/barcode plot from the 05C analysis workflow.

## 3.3 Conservation and Entropy Landscape
Conservation and entropy summaries further reinforced the site-category asymmetry.

### Conservation statistics
- Lassa: mean 0.6163, median 0.5767, SD 0.2039, IQR 0.4332–0.7709
- Ebola: mean 0.9975, median 1.0000, SD 0.0390, IQR 0.9994–1.0000

### Entropy statistics (bits)
- Lassa: mean 1.5078, median 1.6207, SD 0.7051, IQR 1.1681–2.0325
- Ebola: mean 0.0078, median 0.0000, SD 0.0372, IQR 0.0000–0.0075

These values indicate that Ebola positions are near-fixed in this cohort, while Lassa positions retain broader diversity. The joint conservation–entropy profile is therefore consistent with the observed categorical structure.

**[Figure 5 here: Conservation and entropy distributions]**  
Recommended: violin/distribution plots and normalized-position conservation–entropy overlays from the 05C workflow.

## 3.4 Substitution Realization Burden
Observed substitution burden showed a major cross-virus contrast:
- Lassa: 4,128 observed vs 5,692 unobserved substitutions (observed fraction 0.4204)
- Ebola: 255 observed vs 12,589 unobserved substitutions (observed fraction 0.0199)

Thus, in the current dataset composition, Lassa occupies a much broader realized substitution space than Ebola.

## 3.5 Embedding-Space and Outlier Characteristics
Embedding-level comparison (`embedding_comparison_stats.csv`) showed:
- Embedding dimension: 1,280 (both cohorts)
- Norm mean: Lassa 9.6781, Ebola 9.7773
- Norm SD: Lassa 0.1382, Ebola 0.0102
- Centroid Euclidean distance: 1.5857
- Centroid cosine distance: 0.013205
- Separation ratio: 0.9940

Outlier-score analysis (`outlier_score_comparison.csv`) showed:
- Lassa: mean 51.75, SD 9.40, high outliers (>80) = 13
- Ebola: mean 56.85, SD 16.61, high outliers (>80) = 199

Although centroid-level separation was not large by this ratio, Ebola demonstrated a heavier high-outlier tail, indicating stronger atypicality concentration within a subset of sequences.

**[Figure 6 here: PCA and outlier distributions]**  
Recommended: PCA comparison panel plus outlier histogram/violin panel from 05C outputs.

## 3.6 Supervised Classification Performance
Two models were evaluated on the same feature representation:
- Logistic Regression (scaled)
- Random Forest

The selected deployment model was logistic regression (tied best metrics with lower operational complexity). On the held-out split:
- Accuracy = 1.000
- Precision = 1.000
- Recall = 1.000
- F1 = 1.000
- ROC-AUC = 1.000
- Training samples = 1,912
- Test samples = 478

These values indicate complete separation on the current split. However, because perfect performance can arise from cohort-specific structure, external benchmarking remains essential before generalization claims.

## 3.7 Deployment-Ready Inference Outputs
The deployed application (`https://mutation-analysis.streamlit.app`) transformed model outputs into user-facing interpretation by returning, per sequence:
- predicted virus class,
- confidence and probability,
- mutation risk score and risk category,
- atypicality z-score,
- narrative interpretation text,
- report-card and downloadable report artifacts.

This establishes practical translational utility beyond static analysis tables.

## 3.8 Recommended Manuscript Tables and Figures
To keep the Results section concise and publication-ready, structure display items as follows:

- **Table 1**: Cohort and reference-position summary (Lassa vs Ebola).
- **Table 2**: Site-category fractions and counts (`site_category_comparison.csv`).
- **Table 3**: Conservation/entropy statistics (`conservation_entropy_summary_stats.csv`).
- **Table 4**: Observed substitution burden (`observed_substitution_comparison.csv`).
- **Table 5**: Embedding and outlier summaries (`embedding_comparison_stats.csv`, `outlier_score_comparison.csv`).
- **Table 6**: Model leaderboard and selected-model metrics (`training_metrics.json`).

- **Figure 4**: Site-category distribution and barcode track.
- **Figure 5**: Conservation/entropy distributions and normalized overlays.
- **Figure 6**: PCA and outlier profile comparison.

This presentation preserves full quantitative rigor while keeping the main text journal-length compliant.
