# 4.0 Discussion, Benchmarking, and Novelty

## 4.1 Discussion of Core Findings
The principal comparative finding is a pronounced constraint asymmetry between viruses in the analyzed protein contexts. Ebola showed near-global Critical-site dominance (98.96%), while Lassa retained substantial Hotspot and Intermediate fractions (34.83% and 40.73%, respectively). This signal is reinforced by conservation/entropy summaries and by observed substitution burden (Lassa 42.04% versus Ebola 1.99% observed candidate substitutions), supporting a coherent interpretation of broader mutational flexibility in Lassa and tighter mutational tolerance in Ebola.

At representation level, embedding centroid separation was moderate (separation ratio 0.994), but outlier behavior was non-uniform: Ebola displayed a heavier high-outlier tail (n>80: 199 vs 13). This suggests that centroid summaries alone understate local heterogeneity and that atypicality distributions remain informative for comparative interpretation.

## 4.2 Benchmark Positioning Relative to Prior Work
Most prior computational studies in this domain prioritize either (i) mutation/constraint analysis or (ii) predictive modeling, typically as notebook-only outputs. In contrast, this project links comparative virology outputs to a deployable, user-facing sequence interpretation tool.

Benchmark positioning can therefore be stated on three axes:
1. **Comparative scope**: explicit Lassa–Ebola side-by-side quantification in one pipeline.
2. **Translational continuity**: same codebase from analysis tables/figures to deployable app.
3. **Interpretability depth**: prediction + risk category + atypicality + narrative explanation/report export.

This is a methodological benchmark (pipeline integration and usability) rather than a definitive biological-performance benchmark across external cohorts.

## 4.3 Novelty Statement (Journal-Safe Wording)
A defensible novelty claim is:

> To our knowledge, this work is among the first openly deployable frameworks that combines Lassa–Ebola comparative mutation-landscape quantification with sequence-level narrative inference and interpretable risk reporting in a single reproducible pipeline.

This wording avoids overclaiming (“first ever”) while clearly stating the integrated contribution.

## 4.4 Strengths
Key strengths include:
- complete provenance from source FASTA links through curated outputs,
- convergent comparative metrics (site category, conservation/entropy, substitution burden, embedding/outlier behavior),
- transparent low-complexity model suitable for cloud deployment,
- direct practical interface (`https://mutation-analysis.streamlit.app`) for sequence upload and reporting.

## 4.5 Limitations
Important limitations are:
1. Perfect held-out classification metrics indicate strong separability but require external validation to confirm generalization.
2. Composition/length features trade structural depth for interpretability and speed.
3. Risk scoring is outlier-derived and should be treated as triage-oriented, not clinical-grade probability of pathogenicity.
4. Inference is contingent on current curation and may shift with expanded sequence diversity.

## 4.6 Implications and Next Benchmark Steps
For translational strengthening before journal submission, we recommend:
- external validation on independent Lassa and Ebola sequence cohorts,
- temporal/geographic split benchmarking,
- ablation comparison (composition-only vs embedding-enhanced predictors),
- calibration analysis (e.g., Brier score, reliability curves),
- benchmarking against established mutation-effect tools where matched labels exist.

## 4.7 Figure/Table Placement for Discussion Section
- **Figure 7**: Integrated benchmark matrix (prior-work pattern vs this framework).
- **Figure 8**: Practical deployment/interpretability workflow snapshot.
- **Table 5**: Novelty and benchmark dimensions (scope, interpretability, deployability, reproducibility).

## 4.8 Concluding Interpretation
The combined evidence supports a robust comparative distinction between Lassa and Ebola mutational architectures and demonstrates that these insights can be operationalized into an interpretable deployment. This combination of comparative rigor and deployment readiness constitutes the primary translational contribution of the work.
