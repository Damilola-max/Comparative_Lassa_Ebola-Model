# Supplementary Notes: Benchmark and Novelty Support

This file provides supporting material for Section **4.0 Discussion, Benchmarking, and Novelty** in `03_Discussion.md`.

## S1. Compact Benchmark Matrix

| Dimension | Typical Pattern in Prior Work | Current Study |
|---|---|---|
| Scope | Single-virus or non-deployable analyses | Integrated Lassa–Ebola comparative + deployable app |
| Output | Static figures/tables | Static outputs + user-facing inference/report card |
| Interpretability | Expert-centric metrics | Narrative interpretation + risk category + atypicality |
| Reproducibility | Variable artifact availability | Versioned model, metrics, comparative tables, hosted app |

## S2. Source Provenance Registry

### Ebola data anchors
- `https://github.com/ebov/space-time/blob/master/Data/Makona_1610_genomes_2016-06-23.fasta`
- `https://nextstrain.org/ebola/ebov-2013`
- `https://raw.githubusercontent.com/Damilola-max/Comparative_Lassa_Ebola-Model/main/data/raw/Ebola_Protein_Sequence.fas`

### Lassa data anchor
- `https://raw.githubusercontent.com/Damilola-max/Comparative_Lassa_Ebola-Model/main/data/raw/S_protein.fas`

## S3. Journal-Safe Novelty Wording
Recommended wording:

> To our knowledge, this work is among the first openly deployable frameworks integrating Lassa–Ebola comparative mutation-landscape quantification with sequence-level narrative inference and interpretable risk reporting in one reproducible pipeline.

## S4. Extended Benchmark Plan (Revision-Ready)
If reviewers request stronger benchmarking, add:
1. External validation cohorts (geographically/temporally distinct).
2. Feature ablation (composition-only vs embedding-augmented).
3. Additional model families (e.g., gradient boosting, SVM) and calibration metrics.
4. Mutation-effect benchmarking against curated experimental labels where available.

## S5. Citation Anchors (to format per target journal)
1. Rives A, et al. *PNAS* (2021).  
2. Lin Z, et al. *Science* (2023).  
3. Jumper J, et al. *Nature* (2021).  
4. Meier J, et al. zero-shot mutation effect prediction (preprint/peer-reviewed version).  
5. Domain-specific Lassa/Ebola evolutionary constraint literature to be added per target journal scope.

## S6. Repository Citation Entries
- Repository: `https://github.com/Damilola-max/Comparative_Lassa_Ebola-Model/`
- Deployed app: `https://mutation-analysis.streamlit.app`
- Comparative outputs: `results/05C_Result/05C_table/`
- Trained model and metrics: `models/final/`
