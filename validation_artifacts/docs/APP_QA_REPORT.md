# App QA Report and Screenshot Log (R&R)

## App Under Test
- File: `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/app.py`
- Runtime command: `streamlit run app.py`

## QA Objective
Address reviewer concern on deployment robustness by documenting deterministic regression checks, edge-case behavior, and known limitations.

## Test Artifacts Generated
### Tables
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/tables/qa_reference_predictions.csv`
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/tables/qa_edge_case_predictions.csv`

### Screenshot-style PNG summaries
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/figures/qa_reference_predictions.png`
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/figures/qa_edge_case_predictions.png`

## Regression Panel Results
### Canonical references
- Lassa reference is predicted as Lassa (high confidence).
- Ebola reference is predicted as Ebola (high confidence).
- Both receive high/maximum `mutation risk` category under current scale, which supports reviewer concern that current category labels may be misleading for end users.

### Edge-case inputs
- Very short, ambiguous, and noisy sequences produce extreme scores (often `Critical`) with very high confidence values.
- This indicates current post-processing and score mapping do not sufficiently penalize low-information or malformed inputs.

## Identified Deployment Risks
1. **Terminology risk**: UI currently presents `mutation risk` categories (`Harmless`–`Critical`) that may be interpreted as biological/clinical truth.
2. **Input robustness risk**: low-quality inputs can still produce overconfident outputs.
3. **Calibration risk**: atypicality-derived score is repurposed as `risk` without independent biological calibration.

## Immediate App Fixes Recommended
1. Rename output labels in UI/API:
   - `mutation_risk_score` → `atypicality_index`
   - `mutation_risk_category` → `atypicality_band`
2. Add hard validation gates:
   - minimum cleaned length threshold,
   - minimum proportion of canonical residues,
   - explicit warning and abstain behavior for low-quality input.
3. Add visible disclaimer in app header and report export:
   - research-use computational classifier,
   - not a diagnostic or clinical decision tool.
4. Add test automation for reference and edge-case panel in CI.

## Conclusion
The QA checks confirm the reviewer’s robustness concern is valid and addressable. The app can be retained if interpretation is reframed to atypicality, low-quality input handling is hardened, and regression tests are institutionalized.
