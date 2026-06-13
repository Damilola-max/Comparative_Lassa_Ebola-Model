# Comparative Mutation-Landscape Analysis and Deployable Sequence Interpretation for Lassa and Ebola

## Abstract
Lassa and Ebola viruses remain high-priority pathogens with distinct evolutionary trajectories and public-health implications. We developed an end-to-end computational framework that couples comparative mutation-landscape analysis with deployable sequence-level inference. The study integrates curated sequence processing, site-category profiling, conservation/entropy analysis, substitution burden analysis, embedding-space characterization, and supervised classification.

The cleaned analytical cohort included 2,390 sequences (780 Lassa, 1,610 Ebola). Comparative outputs showed major cross-virus asymmetry: Ebola displayed dominant Critical-site occupancy (98.96%), whereas Lassa showed substantial Hotspot (34.83%) and Intermediate (40.73%) fractions. Conservation and entropy summaries aligned with this pattern (mean conservation: Lassa 0.616, Ebola 0.997; mean entropy: Lassa 1.508, Ebola 0.0078 bits). Observed substitution burden similarly diverged (Lassa 42.04% observed candidates vs Ebola 1.99%).

A lightweight classifier (logistic regression) selected for deployment achieved perfect held-out metrics on the current split (accuracy, precision, recall, F1, ROC-AUC all 1.00), and was integrated with an interpretable risk layer (risk score/category plus atypicality z-score) in a public Streamlit app. The resulting pipeline provides both comparative virology insight and practical translational utility for rapid sequence triage. External validation is required to confirm generalization beyond the present cohort.

**Keywords:** Lassa virus, Ebola virus, mutation landscape, ESM embeddings, comparative virology, interpretable machine learning, sequence triage

---

## 1.0 Introduction
Lassa and Ebola viruses are associated with severe outbreaks and substantial clinical burden, motivating improved computational tools for sequence surveillance and mutation interpretation. While many studies either characterize sequence evolution or build predictive models, fewer provide an integrated path from comparative analysis to deployable, user-facing interpretation.

Protein-sequence language models and high-throughput computational virology pipelines have expanded capabilities for mutation profiling, but translational adoption often remains limited by fragmented workflows, limited interpretability, or absence of deployment-ready interfaces. To address this gap, we designed a reproducible framework that unifies comparative mutation-landscape quantification and real-time sequence-level interpretation.

This work asks three core questions:
1. Do Lassa and Ebola exhibit distinct site-level constraint structures in the analyzed protein contexts?
2. Do conservation, entropy, substitution burden, and embedding/outlier analyses converge on the same comparative signal?
3. Can these findings be operationalized into an interpretable, publicly deployable prediction system?

---

## 2.0 Methodology

### 2.1 Study Design
The framework was built as an end-to-end pipeline with two linked components:  
(i) comparative mutation-landscape analysis across Lassa and Ebola; and  
(ii) deployable sequence-level classification with interpretable risk reporting.

### 2.2 Data Sources and Provenance
Data provenance was explicitly tracked in project notebooks and repository workflows.

**Ebola anchors:**
- `https://github.com/ebov/space-time/blob/master/Data/Makona_1610_genomes_2016-06-23.fasta`
- `https://nextstrain.org/ebola/ebov-2013`
- `https://raw.githubusercontent.com/Damilola-max/Comparative_Lassa_Ebola-Model/main/data/raw/Ebola_Protein_Sequence.fas`

**Lassa anchor:**
- `https://raw.githubusercontent.com/Damilola-max/Comparative_Lassa_Ebola-Model/main/data/raw/S_protein.fas`

After cleaning and validation, the unified dataset (`data/cleaned/cleaned_sequences.csv`) contained 2,390 sequences (780 Lassa, 1,610 Ebola).

### 2.3 Sequence Cleaning and Feature Engineering
Sequences were normalized to uppercase and filtered to canonical amino-acid symbols (`ACDEFGHIKLMNPQRSTVWY`) using deterministic cleaning logic in `src/features/sequence_features.py`. For supervised classification, features included sequence length and normalized amino-acid composition.

### 2.4 Comparative Mutation-Landscape Analysis
Comparative outputs were generated in the 05C stage and summarized in:
- `results/05C_Result/05C_table/site_category_comparison.csv`
- `results/05C_Result/05C_table/conservation_entropy_summary_stats.csv`
- `results/05C_Result/05C_table/observed_substitution_comparison.csv`
- `results/05C_Result/05C_table/embedding_comparison_stats.csv`
- `results/05C_Result/05C_table/outlier_score_comparison.csv`

Analyses included site-category structure, conservation/entropy distributions, substitution realization burden, embedding centroid comparison, and outlier-score profiling.

### 2.5 Predictive Modeling and Inference
Two supervised baselines were trained (`src/models/train.py`): logistic regression (scaled) and random forest. A stratified 80/20 split (`RANDOM_STATE=42`) was used, and model selection was based on F1 score.

Inference outputs (`src/models/predict.py`) included:
- predicted class,
- confidence and class probability,
- mutation risk score (0–100),
- risk category,
- atypicality z-score.

### 2.6 Deployment and Reproducibility
The app is publicly deployed at `https://mutation-analysis.streamlit.app` and implemented in `app.py`. Reproducible execution:

```bash
pip install -r requirements.txt
python3 scripts/03_train.py
python3 scripts/04_evaluate.py
streamlit run app.py
```

---

## 3.0 Results

### 3.1 Cohort and Output Summary
The final cohort comprised 2,390 sequences. Comparative and model outputs were generated reproducibly and persisted in versioned repository artifacts.

### 3.2 Site-Category Asymmetry
Site-category analysis demonstrated profound cross-virus asymmetry:
- Ebola: 669/676 Critical sites (98.96%); no Hotspot or Intermediate sites.
- Lassa: 2/491 Critical sites (0.41%); Hotspot 34.83%; Intermediate 40.73%.

This indicates substantially tighter positional constraint in Ebola and broader positional flexibility in Lassa.

**[Figure 4: Site-category distribution + barcode tracks]**

### 3.3 Conservation and Entropy
Conservation/entropy summaries reinforced this pattern:
- Mean conservation: Lassa 0.6163 vs Ebola 0.9975.
- Mean entropy: Lassa 1.5078 bits vs Ebola 0.0078 bits.

Ebola sites were near-fixed across most positions; Lassa retained broader sequence heterogeneity.

**[Figure 5: Conservation and entropy distributions/overlays]**

### 3.4 Substitution Burden
Observed substitution fractions:
- Lassa: 4,128 observed out of 9,820 candidates (42.04%).
- Ebola: 255 observed out of 12,844 candidates (1.99%).

This supports a markedly narrower realized substitution envelope in Ebola.

### 3.5 Embedding and Outlier Structure
Embedding and outlier summaries showed:
- Centroid Euclidean distance = 1.5857
- Centroid cosine distance = 0.013205
- Separation ratio = 0.9940
- Outlier high-tail burden (>80): Lassa 13 vs Ebola 199

Despite moderate centroid-level separation, Ebola exhibited a heavier high-atypicality tail.

**[Figure 6: PCA + outlier distribution comparison]**

### 3.6 Supervised Classification
Logistic regression was selected for deployment (tied best metrics, lower operational complexity). Held-out metrics:
- Accuracy = 1.000
- Precision = 1.000
- Recall = 1.000
- F1 = 1.000
- ROC-AUC = 1.000
- Train/Test = 1,912 / 478

These results suggest strong split-level separability; independent validation is required for generalization claims.

### 3.7 Deployable Interpretation Output
The deployed app returns sequence-level predictions with confidence, risk score/category, atypicality, narrative interpretation, and downloadable report outputs.

---

## 4.0 Discussion, Benchmarking, and Novelty
The comparative findings consistently indicate divergent mutational architectures in the analyzed proteins: Ebola is dominated by Critical-site structure, while Lassa presents broader hotspot/intermediate variability. This convergence across site categories, conservation/entropy, and substitution burden strengthens biological interpretability.

At representation level, the moderate separation ratio with substantial Ebola high-outlier burden suggests that local atypicality behavior remains informative even when centroid summaries are not strongly separative.

From a benchmarking perspective, this work contributes a methodological integration not commonly present in prior single-stage studies: comparative quantification, model inference, and public deployment are provided in one reproducible pipeline. The key translational contribution is not only predictive performance, but interpretation accessibility through narrative reports and risk stratification.

A journal-safe novelty statement is:

> To our knowledge, this study is among the first openly deployable frameworks integrating Lassa–Ebola comparative mutation-landscape quantification with sequence-level narrative inference and interpretable risk reporting in a single reproducible workflow.

### Limitations
1. Perfect held-out metrics likely reflect strong cohort separability and require external validation.
2. Composition-focused features prioritize deployability/interpretability over structural richness.
3. Risk scores are triage-oriented outlier-derived signals, not clinical risk probabilities.
4. Findings remain contingent on current cohort composition and curation scope.

### Recommended next benchmarks
- External validation on independent sequence cohorts.
- Temporal/geographic split evaluation.
- Feature-ablation and embedding-augmented model comparison.
- Calibration analysis and uncertainty quantification.

---

## 5.0 Conclusion
This study provides an integrated, reproducible, and deployable framework for comparative Lassa–Ebola mutation analysis and sequence-level interpretation. The evidence supports substantial cross-virus asymmetry in mutational constraint architecture and demonstrates that these insights can be translated into practical inference outputs for rapid sequence triage.

---

## Data and Code Availability
- Repository: `https://github.com/Damilola-max/Comparative_Lassa_Ebola-Model/`
- Deployed app: `https://mutation-analysis.streamlit.app`
- Comparative tables: `results/05C_Result/05C_table/`
- Trained model and metrics: `models/final/`

---

## References (To Format Per Target Journal)
1. Rives A, et al. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. *PNAS* (2021).  
2. Lin Z, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* (2023).  
3. Jumper J, et al. Highly accurate protein structure prediction with AlphaFold. *Nature* (2021).  
4. Related mutation-effect language-model studies (final citation to be harmonized with journal style).  
5. Domain-specific Lassa and Ebola evolutionary/constraint studies (final selection per target journal scope).
