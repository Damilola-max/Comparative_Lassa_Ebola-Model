# Test Protocol (R&R)

## Purpose
Provide a repeatable protocol for validation, ablation, and deployment QA checks tied to reviewer concerns.

## Preconditions
- Repository root: `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model`
- Python environment with `requirements.txt` installed.

## Mandatory checks
1. Security scan:
   - `snyk test --all-projects`
2. Lint checks:
   - `ruff check .`
   - `ruff check . --fix`

## Validation and ablation execution
Run:
- `python3 scripts/rnr_revision_analysis.py`

Expected output:
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/revision_analysis_summary.json`
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/tables/*.csv`
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/results/rnr_revision/figures/*.png`

## QA panel checks
### Reference panel
Input files:
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/data/Reference/Lassa_Reference_Sequence.fasta`
- `/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/data/Reference/Ebola_Reference_Sequence.fasta`

Acceptance criteria:
- Lassa reference predicted as Lassa.
- Ebola reference predicted as Ebola.
- Output fields present: predicted label, confidence, atypicality score/band.

### Edge-case panel
Inputs:
- very short sequence (`ACD`)
- ambiguous/noisy sequence (`AXXZ---TT??`)
- numeric/noisy sequence (`A12345VVVV`)
- long synthetic repeat sequence

Acceptance criteria:
- App should not crash.
- App should return explicit quality warning or abstain behavior (recommended update).

## Regression artifacts to archive
- `results/rnr_revision/tables/qa_reference_predictions.csv`
- `results/rnr_revision/tables/qa_edge_case_predictions.csv`
- `results/rnr_revision/figures/qa_reference_predictions.png`
- `results/rnr_revision/figures/qa_edge_case_predictions.png`

## Reporting
Summarize in:
- `manuscript/SUPPLEMENTARY_METHODS_VALIDATION_REPORT.md`
- `manuscript/APP_QA_REPORT.md`

Include:
- exact command logs,
- key metric tables,
- caveats and limitations,
- interpretation boundaries.
