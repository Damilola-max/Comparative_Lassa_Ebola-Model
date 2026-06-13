# Response to Reviewers (R&R Draft)

## Editorial Recommendation
We appreciate the recommendation for major revision and have revised claims, strengthened validation reporting, expanded dataset transparency artifacts, and documented deployment QA limitations.

---

## Reviewer Concern 1
**Concern:** Biological implications extend beyond analyzed datasets.

**Response:**
- We moderated interpretation language and added explicit dataset-bounded framing in Discussion.
- Updated manuscript now states that extrapolation requires additional external validation and biological follow-up.

**Evidence files:**
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/manuscript/FULL_MANUSCRIPT.md` (Discussion section, added calibration paragraph)

---

## Reviewer Concern 2
**Concern:** Dataset composition and exact proteins are insufficiently described.

**Response:**
- Added machine-generated dataset manifest and summary tables with sequence-level metadata parsing, composition counts, and duplicate statistics.
- Added explicit note that some metadata fields are unknown for subsets and should not be over-interpreted.

**Evidence files:**
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/tables/dataset_manifest.csv`
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/tables/dataset_virus_summary.csv`
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/tables/dataset_missingness.csv`
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/tables/duplicate_summary.csv`
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/manuscript/SUPPLEMENTARY_METHODS_VALIDATION_REPORT.md`

---

## Reviewer Concern 3
**Concern:** ESM-2 is described as central although deployed classifier appears composition-based.

**Response:**
- We ran ablation analysis (composition-only, ESM-only, composition+ESM).
- Results show composition-only already provides near-perfect discrimination for this task; ESM is not uniquely decisive for classifier performance in this setup.
- We therefore recommend moderating ESM-centrality claims for the deployed classifier.

**Evidence files:**
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/tables/ablation_summary.csv`
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/figures/ablation_f1_logistic.png`

---

## Reviewer Concern 4
**Concern:** Perfect classification may be due to separability, not true predictive ability.

**Response:**
- Added repeated stratified CV, grouped CV, and external grouped holdout analyses.
- Near-perfect metrics persisted, supporting the interpretation that the current label task is highly separable in this dataset.
- We now frame this as high separability rather than proof of broad biological predictive generalization.

**Evidence files:**
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/tables/validation_summary.csv`
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/tables/external_group_holdout_metrics.csv`
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/figures/validation_split_comparison.png`

---

## Reviewer Concern 5
**Concern:** Mutation-risk scoring framework is not biologically validated; terminology may be misleading.

**Response:**
- We revised manuscript terminology toward computational atypicality scoring in title/method sections and added stronger interpretation boundaries.
- We documented that current score bands are model-internal and not clinically validated risk probabilities.

**Evidence files:**
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/manuscript/FULL_MANUSCRIPT.md`
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/manuscript/SUPPLEMENTARY_METHODS_VALIDATION_REPORT.md`

---

## Reviewer Concern 6
**Concern:** Deployed app showed misclassifications/unrealistic risk assignments; robustness concern.

**Response:**
- Added QA regression panel outputs for canonical references and edge cases.
- Documented robustness limitations and required app hardening steps (validation gates, disclaimers, terminology cleanup, CI regression tests).

**Evidence files:**
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/tables/qa_reference_predictions.csv`
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/tables/qa_edge_case_predictions.csv`
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/figures/qa_reference_predictions.png`
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/figures/qa_edge_case_predictions.png`
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/manuscript/APP_QA_REPORT.md`
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/manuscript/TEST_PROTOCOL.md`

---

## Additional Deliverables Included
- Supplementary methods + validation report:
  - `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/manuscript/SUPPLEMENTARY_METHODS_VALIDATION_REPORT.md`
- App QA report:
  - `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/manuscript/APP_QA_REPORT.md`
- Test protocol:
  - `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/manuscript/TEST_PROTOCOL.md`

---

## Remaining Work Before Final Resubmission
To fully satisfy the strongest form of biological-validation critique, add at least one direct biology-linked validation layer:
1. domain/epitope mapping of high-scoring variants,
2. comparison against known fitness/escape mutation sets,
3. curated independent variant test set with biological annotations.
