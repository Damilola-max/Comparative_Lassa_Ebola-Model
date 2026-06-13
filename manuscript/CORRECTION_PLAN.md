# ESM-embedR Manuscript — Step-by-Step Correction Plan

## Overview

This document lists every correction needed to address reviewer concerns, mapped to exact locations in `manuscript/Comparative_Analysis_Refined_3_1.md`. Each correction shows the **current text**, the **proposed change**, and the **expected output**. Read through, then approve each correction block before I execute it.

---

## Correction 1: Title — Remove "Operational Risk Stratification"

**Reviewer concern:** The title implies clinical/operational risk assessment that has not been biologically validated.

**Current (line 1):**
```
***ESM-embedR: A Protein Language Model Framework for Comparative Mutation Analysis and Operational Risk Stratification of Lassa and Ebola Virus Sequences***
```

**Proposed:**
```
***ESM-embedR: A Protein Language Model Framework for Comparative Mutation Analysis and Computational Atypicality Scoring of Lassa and Ebola Virus Sequences***
```

**Expected output:** Title uses neutral computational language.

---

## Correction 2: Abstract — Moderate "risk stratification" terminology

**Reviewer concern:** Abstract frames the output as clinical risk stratification.

**Current (line 14, truncated):**
> ...supervised machine learning classification with interpretable risk stratification.

**Proposed:**
> ...supervised machine learning classification with interpretable computational atypicality scoring.

**Also in Abstract (same paragraph):**
> ...generating mutation risk scores, categorized risk levels...

**Proposed:**
> ...generating a computational atypicality index with bounded interpretive bands...

**Expected output:** Abstract describes the tool as a computational atypicality framework, not a risk predictor.

---

## Correction 3: Methods Section — Rename "Risk Stratification" to "Atypicality Scoring"

**Reviewer concern:** Methods section title uses unvalidated risk terminology.

**Current (line 56):**
```
### Supervised Classification and Risk Stratification
```

**Proposed:**
```
### Supervised Classification and Atypicality Scoring
```

**Expected output:** Methods heading reflects the actual computational nature of the scoring.

---

## Correction 4: Methods — Clarify that classifier uses composition, not ESM-2 embeddings

**Reviewer concern:** The manuscript implies ESM-2 embeddings are central to the classifier, but the deployed model uses 21-dimensional amino-acid composition features.

**Current (lines 58-60):**
> Each sequence was featurized by computing its length (seq_length) and the normalized frequency of each canonical amino acid (aa_freq_A through aa_freq_Y), yielding a 21-dimensional feature vector per sequence.

**This text is actually correct** — but the surrounding narrative (Introduction, Abstract) implies ESM-2 drives classification. We need an explicit clarification paragraph.

**Proposed insertion after current feature description (after line 60):**
> **Clarification on ESM-2 role.** ESM-2 embeddings (1,280-dimensional) were generated for comparative embedding-space analysis (centroid distances, outlier detection, PCA visualization) as described in the "Comparative Mutation-Landscape Analysis" subsection. However, the supervised classifier described here operates on lightweight, interpretable composition features (sequence length + amino-acid frequencies) rather than ESM-2 embeddings. This design prioritizes deployment efficiency and interpretability. An ablation study comparing composition-only, ESM-2-only, and hybrid feature spaces is reported in Supplementary Analysis 1.

**Expected output:** Reader understands ESM-2 = comparative analysis layer; composition features = classification layer.

---

## Correction 5: Methods — Replace "mutation risk score" with "atypicality index"

**Reviewer concern:** The scoring formula is described as "risk" but is actually a statistical deviation from training centroids.

**Current (lines 60, risk formula):**
> ...the mutation risk score was derived as risk_score = 50.0 + 15.0 × z with the result clamped to the interval [0.0, 100.0]. This risk score was mapped to five discrete categories using fixed thresholds, namely Harmless for scores below 20, Neutral for scores from 20 to 39, Moderate for scores from 40 to 59, Dangerous for scores from 60 to 79, and Critical for scores of 80 or above.

**Proposed:**
> ...a bounded atypicality index was derived as index = 50.0 + 15.0 × z, clamped to [0.0, 100.0], and mapped to five fixed interpretive bands for interface readability: Low (<20), Below-Average (20–39), Average (40–59), Elevated (60–79), and High (≥80). **Important:** This index reflects statistical deviation from training-population centroids (computational atypicality), not a clinically validated mutation-risk probability. Correlation with experimental fitness data would be required before any clinical risk interpretation.

**Expected output:** The scoring is framed as a model-internal deviation metric with explicit interpretive boundaries.

---

## Correction 6: Methods — Add explicit mathematical formulas

**Reviewer concern:** Formulas are embedded in prose but not presented as explicit equations.

**Proposed insertions:**

**After line 52 (Conservation definition):** Add equation block:
```
$$C_i = \max_j p_{ij}$$
```

**After line 52 (Shannon entropy):** Add equation block:
```
$$H_i = -\sum_j p_{ij} \log_2(p_{ij})$$
```

**After line 54 (Substitution burden):** Add equation block:
```
$$B_i = \frac{\text{observed substitutions at } i}{380}$$
```

**After line 54 (Centroid):** Add equation block:
```
$$\mu_c = \frac{1}{N_c} \sum_{k=1}^{N_c} x_k$$
```

**After line 60 (Atypicality z-score):** Add equation block:
```
$$z = \frac{\|x - \mu_c\|_2 - \bar{d}_c}{\sigma_c}$$
```

**After line 60 (Bounded index):** Add equation block:
```
$$I = \text{clip}(50 + 15z, 0, 100)$$
```

**Expected output:** Methods section contains clearly formatted LaTeX equations.

---

## Correction 7: Methods — Add dataset transparency table

**Reviewer concern:** Dataset composition (accession IDs, exact proteins, lengths, hosts, dates, lineages) is insufficiently described.

**Current (lines 42-44):** Only mentions source URLs and final counts.

**Proposed insertion after line 44:**
```
### Dataset Composition and Metadata Summary

The final analytical cohort comprises 2,390 sequences. Table D1 provides a virus-level summary. Full sequence-level metadata (including parsed accession IDs, host, country, collection date, and lineage where available from sequence headers) is provided in Supplementary Table 1. Some metadata fields are incomplete in source records and are reported as "unknown" rather than imputed.

**Table D1. Cohort Summary by Virus**

| Virus | N Sequences | Protein | Mean Length | Median Length | Accession Coverage | Known Lineages |
|-------|-------------|---------|-------------|---------------|--------------------|----------------|
| LASV  | 780         | S (GP)  | [value]     | [value]       | Partial            | I–VII          |
| EBOV  | 1,610       | [name]  | [value]     | [value]       | Full (Makona)      | Makona         |
```

**Expected output:** A dataset transparency subsection with a summary table and reference to supplementary metadata.

---

## Correction 8: Methods — Add duplicate-sequence handling disclosure

**Reviewer concern:** Exact duplicate sequences (107 total: 101 EBOV, 6 LASV) were identified in the revision analysis but not mentioned in the original manuscript.

**Proposed insertion in Preprocessing subsection (after line 48):**
```
**Duplicate handling.** Post-cleaning inspection identified 107 exact duplicate sequences globally (101 EBOV, 6 LASV). These were retained in the comparative and embedding analyses because they reflect genuine sampling redundancy in public repositories, but their presence means that the effective independent sequence count is slightly lower than the total sequence count. For classification, the stratified train-test split was applied after deduplication awareness; the reported training/test counts (1,912 / 478) are post-split tallies that include duplicates present in the original cohort.
```

**Expected output:** Transparent disclosure of duplicate presence and handling rationale.

---

## Correction 9: Results — Frame perfect classification as dataset separability

**Reviewer concern:** "Perfect" metrics are presented as proof of model quality, but likely reflect trivial class separability (different viruses with different proteins).

**Current (line 191-193):**
> Both evaluated models achieved perfect discrimination on the held-out test set.

**Proposed:**
> Both evaluated models achieved perfect discrimination on the held-out test set from this cohort. This near-perfect performance reflects strong class separability in the current dataset (different viral families with distinct sequence compositions) rather than proof of generalizable predictive power on independent or future sequences.

**Current (line 367 / Table 6):**
> Logistic regression was selected for deployment due to equivalent performance with lower operational complexity.

**Proposed addition after that sentence:**
> The perfect metrics should be interpreted cautiously: they indicate that the current LASV and EBOV sequences are readily separable using composition features, not that the model will generalize perfectly to all future LASV/EBOV variants or to other filoviruses/arenaviruses.

**Expected output:** Results section calibrates perfect metrics with appropriate caveats.

---

## Correction 10: Results — Add ablation study reference

**Reviewer concern:** ESM-2 value for the classifier is unproven.

**Proposed insertion after Table 6 (after line 200):**
```
**Ablation analysis.** To assess whether ESM-2 embeddings add value beyond composition features for virus classification, we evaluated three feature sets: (1) composition-only (21-dimensional), (2) ESM-2-only (1,280-dimensional), and (3) composition + ESM-2 concatenated. All three configurations achieved near-perfect classification on this dataset, indicating that composition features alone are sufficient for the LASV vs. EBOV discrimination task. ESM-2 embeddings remain valuable for the comparative embedding-space characterization (outlier detection, centroid geometry) reported in the mutation-landscape analysis, but they do not provide additional discriminative power for the classifier in this specific setup. Full ablation metrics are provided in Supplementary Table 2.
```

**Expected output:** Honest reporting of ESM-2's role: essential for comparative embedding analysis, non-essential for the classifier.

---

## Correction 11: Discussion — Add dataset-bounded framing paragraph

**Reviewer concern:** Biological claims extend beyond the analyzed datasets.

**Proposed insertion at start of Discussion (before line 202):**
```
**Interpretive scope.** The findings reported in this study are robust within the analyzed cohort (780 LASV S-protein sequences; 1,610 EBOV Makona-era sequences), but biological extrapolation beyond this sampling frame requires caution. The LASV sequences represent curated S-protein diversity, and the EBOV sequences are concentrated in the 2013–2016 West African epidemic. Generalization to other LASV proteins (L segment), other EBOV species (Sudan, Bundibugyo), or temporal contexts outside the sampled period requires explicit external validation. Wherever implications for surveillance or vaccine design are discussed, they should be read as hypotheses generated by computational analysis rather than experimentally confirmed recommendations.
```

**Expected output:** Discussion opens with explicit boundaries on generalizability.

---

## Correction 12: Discussion — Soften overconfident causal language

**Reviewer concern:** Phrases like "conclusively demonstrates," "unambiguous conclusion," "fundamentally different protein architecture" overstate certainty.

**Current (line 202-204):**
> The evidence from five independent analytical dimensions converges on a single, unambiguous conclusion: Lassa virus and Ebola virus occupy opposite extremes of the viral mutational-constraint spectrum, and the magnitude of this divergence exceeds most documented inter-species comparisons in RNA virology.

**Proposed:**
> The evidence from five analytical dimensions consistently indicates that, within the analyzed datasets, Lassa virus and Ebola virus occupy markedly different regions of the mutational-constraint spectrum. This divergence is pronounced in the current cohort, though its magnitude relative to other RNA-virus comparisons should be assessed with additional external datasets.

**Current (line 236 / Conclusion):**
> The evidence conclusively demonstrates a profound asymmetry in mutational architecture between Lassa virus and Ebola virus.

**Proposed:**
> The evidence supports a strong asymmetry in mutational architecture between Lassa virus and Ebola virus within the analyzed datasets.

**Expected output:** Certainty language is replaced with calibrated, dataset-bounded phrasing.

---

## Correction 13: Discussion — Reframe "risk" as "atypicality" throughout

**Reviewer concern:** Multiple references to "risk stratification," "risk interpretation," "risk scores" imply clinical validation.

**Current (line 216):**
> ...real-time narrative risk interpretation through a publicly hosted interface.

**Proposed:**
> ...real-time narrative atypicality interpretation through a publicly hosted interface.

**Current (line 222):**
> ...rare atypical variants emerge and merit monitoring.

**This is actually fine** — "merit monitoring" is acceptable as a surveillance hypothesis.

**Current (line 232 / Limitations):**
> The mutation risk scores are atypicality-derived triage indicators, not clinical-grade risk probabilities.

**This is already good** — keep but strengthen.

**Current (line 64 / Deployment):**
> ...renders prediction outputs including virus label, confidence, mutation risk score, risk category badge...

**Proposed:**
> ...renders prediction outputs including virus label, confidence, computational atypicality index, interpretive band badge...

**Expected output:** All deployment and discussion references use "atypicality" rather than "risk."

---

## Correction 14: Conclusion — Add explicit non-clinical disclaimer

**Reviewer concern:** The conclusion sounds like the tool is ready for clinical use.

**Current (line 236):**
> The analytical framework... has been deployed as a publicly accessible application...

**Proposed addition at end of Conclusion:**
> The deployed application is intended for research and educational use in comparative sequence analysis. It is not a clinical decision-support tool, and its outputs should not be used for patient diagnosis, treatment selection, or public-health intervention decisions without independent biological validation.

**Expected output:** Clear boundary between research tool and clinical application.

---

## Correction 15: App/Deployment — Update terminology in app.py

**Reviewer concern:** The deployed app still shows "mutation risk score" and "Critical/Dangerous" labels to end users.

**This is a code change, not a manuscript change**, but it must be done for consistency.

**Proposed changes in `app.py` and `src/models/predict.py`:**
- Rename all `risk_score` variables to `atypicality_index`
- Rename `risk_category` to `atypicality_band`
- Change category labels: Harmless → Low, Neutral → Below-Average, Moderate → Average, Dangerous → Elevated, Critical → High
- Add a visible UI disclaimer: "This index reflects statistical deviation from training data, not a validated clinical risk assessment."

**Expected output:** Deployed app language matches manuscript atypicality framing.

---

## Correction 16: Add Supplementary Materials section

**Reviewer concern:** No supplementary materials document the additional analyses.

**Proposed insertion before References:**
```
## Supplementary Materials

**Supplementary Table 1.** Full dataset manifest with sequence-level metadata (accession ID, host, country, collection date, lineage, sequence length, duplicate flag).

**Supplementary Table 2.** Ablation study results: composition-only vs. ESM-2-only vs. hybrid feature classification metrics.

**Supplementary Table 3.** Cross-validation results: repeated stratified K-fold and grouped K-fold metrics with 95% confidence intervals.

**Supplementary Figure 1.** Validation split comparison plot (stratified vs. grouped CV).

**Supplementary Figure 2.** Ablation F1 comparison plot.

**Supplementary Methods Document.** Detailed description of leakage controls, duplicate handling, QA regression panel, and test protocol.
```

**Expected output:** Manuscript references supplementary tables/figures that already exist in `results/rnr_revision/`.

---

## Summary: Execution Order

Once you approve, I will execute corrections in this order:

1. **Title + Abstract** (Corrections 1–2)
2. **Methods** — section rename, formula insertions, ESM-2 clarification, dataset table, duplicate disclosure (Corrections 3–8)
3. **Results** — perfect metrics framing + ablation (Corrections 9–10)
4. **Discussion** — scope paragraph + language softening + atypicality reframe (Corrections 11–13)
5. **Conclusion** — disclaimer addition (Correction 14)
6. **Supplementary section** — insertion (Correction 16)
7. **Code sync** — `app.py` and `predict.py` terminology (Correction 15)

After each batch, I will show you the diff for approval before proceeding to the next batch.
