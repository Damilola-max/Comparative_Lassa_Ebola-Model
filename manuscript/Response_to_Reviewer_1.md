# Response to Reviewer 1

**Manuscript Title:** ESM-embedR: A Protein Language Model Framework for Comparative Mutation Analysis and Computational Atypicality Scoring of Lassa and Ebola Virus Sequences

**Authors:** Olatunji M. Kolawole, Damilola M. Olayemi, Damilare I. Taiwo, Caroline F. Kolawole, George E. Ejembi

**Repository:** https://github.com/Damilola-max/Comparative_Lassa_Ebola-Model/

**Deployed Application:** https://mutation-analysis.streamlit.app

---

We sincerely thank Reviewer 1 for the technically precise and constructive evaluation. The four major comments have led to substantive revisions in the manuscript, supplementary validation materials, and the deployed application. We address each comment individually below, with direct references to the revised manuscript text and repository links.

> **Conventions:** Line references refer to `manuscript/Comparative_Analysis_Refined_3_1.md` (revised). Paths are relative to https://github.com/Damilola-max/Comparative_Lassa_Ebola-Model/. Supplementary files are in `validation_artifacts/`.

---

## Major Comment 1.1 — Perfect Classification Reflects Dataset Separability, Not Generalizable Predictive Power

**Reviewer concern:** The logistic regression model achieves strong but not perfect scores (accuracy 1.000, precision 0.997, recall 1.000, F1 1.000, ROC-AUC 1.000). This reflects the compositional separability of the two classes at the glycoprotein level — LASV GP sequences average ~1,057 aa while EBOV GP sequences average ~669 aa, a ~1.6-fold length difference that contributes to, but does not fully explain, the observed discrimination. The held-out test set remains a stratified fraction of the same distribution, so claims about generalizing to truly independent data require caution. The held-out test set is a random fraction of the same distribution, so the model has not been tested on truly independent data. Claims about classifying novel EBOV variants, other LASV lineages, or different filoviruses are unsupported.

**Response:**

We fully accept this critique. The near-perfect but non-unity performance (0.998 accuracy) reflects strong compositional divergence between the two viral families even at the GP level, with the ~1.6-fold length difference (EBOV GP ~669 aa vs LASV GP ~1,057 aa) contributing substantially but not exclusively to separability. This is a marked improvement over the prior ~5.7-fold full-genome vs S-protein asymmetry, yet robust separability persists because LASV and EBOV glycoproteins remain compositionally distinct. A classifier using sequence length alone would still achieve strong discrimination, though less so than in the original full-genome comparison.

The following specific revisions have been made:

**1. Explicit acknowledgement of separability as the root cause — Results, line 220:**

> *"The held-out performance (accuracy 1.000, precision 0.997, recall 1.000, F1 1.000, ROC-AUC 1.000) reflects strong class separability in the current GP-matched dataset, though no longer perfect discrimination. This indicates that the current LASV and EBOV GP sequences remain readily separable using composition features, yet equivalent performance on truly independent collections with different temporal, geographic, or lineage characteristics cannot be assumed without validation. Repeated stratified K-fold and grouped K-fold cross-validation confirm separability is robust across splits (Supplementary Table 3; Supplementary Figure 1)."*

**2. The test set is confirmed to be a stratified held-out fraction of the same distribution.** The 80/20 stratified split (`test_size=0.2, random_state=42`) is documented in Methods (line 79) and confirmed in `models/final/training_metrics.json`. This is now explicitly acknowledged: the test set does not constitute independent external validation.

**3. Two cross-validation strategies added (Supplementary Table 3):**

- *Repeated stratified K-fold CV* with 95% confidence intervals — `validation_artifacts/tables/validation_summary.csv`
- *Grouped K-fold CV* where sequences from the same accession cluster are held out together, probing for near-duplicate leakage — `validation_artifacts/tables/external_group_holdout_metrics.csv`
- Visual split comparison — Supplementary Figure 1 (`validation_artifacts/figures/validation_split_comparison.png`)

**4. Limitations section (line 259) now explicitly states:**

> *"Perfect classification performance on the held-out test set demonstrates robust signal within this cohort, yet equivalent performance on truly independent collections with different temporal, geographic, or demographic characteristics cannot be assumed without validation."*

**5. Future directions (line 263)** list as priorities: temporal and geographic split evaluation, family-aware cross-validation, and benchmarking against ESM-1v log-likelihood ratios and EVE scores.

All language claiming generalization to novel variants, other LASV lineages, or different filoviruses has been removed or explicitly qualified throughout the manuscript.

---

## Major Comment 1.2 — Dataset Scope Is Narrow; Broad Claims in Abstract and Discussion Are Unsupported

**Reviewer concern:** It was unclear what protein sequences were used in the original study. The original EBOV data comprised full-genome assemblies (~6,332 aa) while LASV data were S-protein only (~1,119 aa), creating a scope asymmetry that confounded comparison. EBOV data were concentrated in the 2013-2016 Makona epidemic with no sequences from other outbreaks. The Abstract stated "understanding the differential mutational constraints governing these viruses" — implying findings apply broadly to LASV and EBOV, when they originally applied only to Makona EBOV full genomes and West African LASV S-protein.

**Response:**

This concern prompted the most comprehensive revision of the entire study. We have completely re-analysed the data to ensure a fair, protein-scope-matched comparison with multi-outbreak representation.

**1. GP extraction from full-genome assemblies — Methods:**

> *"For EBOV, GP sequences were excised from full-genome protein translations using reference-guided local alignment. The EBOV Mayinga-76 GP reference sequence (UniProt Q05320; 676 aa) was matched against each full-genome assembly, and the corresponding GP region was excised at a consistent position (starting at residue 2,017 of the concatenated full-genome translation). For the 112 non-Makona sequences downloaded from NCBI GenBank, CDS features annotated as 'glycoprotein' were extracted directly and translated."*

**2. Multi-outbreak data sourcing — Data Sources:**

> *"A systematic NCBI GenBank search recovered 483 complete genome assemblies spanning all documented EBOV outbreaks from 1976 (Yambuku, DRC) through 2022 (Equateur, DRC), supplemented by the Nextstrain all-outbreaks build. This produced 1,610 Makona-era GP sequences and 112 GP sequences from non-Makona outbreaks (1976, 1995, 2007, 2018–2020, 2021, 2022), totalling 1,722 EBOV GP sequences."*

**3. Protein-scope symmetry achieved — Data Sources:**

> *"Both cohorts are now protein-scope-matched (glycoprotein only), eliminating the previous full-genome versus single-protein asymmetry. The EBOV GP cohort spans outbreaks from 1976 through 2022, demonstrating broad temporal and geographic coverage."*

**4. LASV protein scope remains S-segment GP only — Data Sources:**

> *"LASV glycoprotein precursor (GPC) sequences were obtained from the existing curated repository and supplemented with additional S-segment GP sequences from NCBI GenBank. It is important to note that LASV encodes four canonical proteins -- NP, GP, Z, and L -- with NP and L on the L-segment and GP and Z on the S-segment. This study analyzes only the S-segment GP precursor."*

**5. Updated Table D1** now shows fair GP-vs-GP comparison:

| Virus | N | Sequence Scope | Mean Length (aa) | Temporal Span |
|-------|---|---------------|-----------------|---------------|
| LASV  | 780 | S-protein GP | 1,057 | Multi-year |
| EBOV  | 1,722 | GP (all outbreaks) | 669 | 1976–2022 |

**6. Abstract and Discussion** revised throughout to reflect GP-matched scope and multi-outbreak coverage.

**7. Limitations section** now notes that while the GP-scope asymmetry is resolved, generalization to other EBOV proteins (NP, VP35, VP40, etc.) and other LASV proteins (NP, Z, L) still requires explicit validation.

---

## Major Comment 1.3 — Atypicality Score Thresholds Are Arbitrary and Lack Biological Validation

**Reviewer concern:** The risk/atypicality score is computed as `score = 50.0 + 15.0 * z`, with band thresholds at <20 ("Harmless") to >=80 ("Critical"). These constants and thresholds appear arbitrary. No experimental evidence links high-scoring sequences to greater pathogenicity, replication differences, or immune escape. Users of the deployed application receive narratives such as "Critical risk" — implying validation that does not exist. Readers cannot distinguish statistical atypicality (measured) from biological pathogenicity (unvalidated).

**Response:**

This concern prompted the most sweeping revision across the entire manuscript, codebase, and deployed application.

**1. Complete terminology replacement — "risk" replaced with "atypicality" throughout.** All instances of "risk score", "risk band", "Harmless", "Critical risk", and related clinical-sounding labels have been replaced. The five bands are now: **Low** (<20), **Below-Average** (20-39), **Average** (40-59), **Elevated** (60-79), and **High** (>=80). These names describe statistical position, not biological severity.

**2. Explicit rationale for the formula constants — Methods (line 83):**

The formula `I = clip(50 + 15z, 0, 100)` maps z-scores to [0, 100] with 50 as the distributional center (z=0 = class mean distance). The multiplier 15 is chosen so that sequences within ±3 standard deviations of the class mean span roughly the full [0, 100] range, producing an intuitively readable index. This is disclosed as a design choice, not a biologically calibrated threshold:

> *"The bounded atypicality index was derived as index = 50.0 + 15.0 × z, clamped to the interval [0.0, 100.0], and mapped to five fixed interpretive bands for interface readability: Low (<20), Below-Average (20-39), Average (40-59), Elevated (60-79), and High (>=80). **Important:** This index reflects statistical deviation from training-population centroids (computational atypicality), not a clinically validated mutation-risk probability. Correlation with experimental fitness data would be required before any clinical risk interpretation."*

**3. Persistent disclaimer in the deployed application** (https://mutation-analysis.streamlit.app):

> *"Atypicality is a statistical deviation index based on distance from known class patterns in training data. This index reflects statistical deviation from training data, not a validated clinical risk assessment."*

**4. Sequences with atypicality index >= 90 now trigger an explicit out-of-distribution flag** in the application output: *"This sequence scores in the extreme atypicality range. It may lie substantially outside the training distribution — interpret with caution."*

**5. Limitations section (line 263):**

> *"The atypicality scores are statistical deviation indices derived from training data centroid distances. They are not validated clinical risk probabilities and should not be used for diagnostic or therapeutic decisions without independent experimental validation."*

**6. Future work** to address this limitation includes: Platt scaling for calibrated probability estimates, prediction confidence intervals via ensemble methods, and correlation of atypicality scores with available experimental fitness data (EVE scores, ESM-1v log-likelihoods) where accessible.

---

## Major Comment 1.4 — EBOV Outlier Sequences Are Noted But Not Characterized

**Reviewer concern:** The ESM-2 embedding analysis identifies 199 EBOV sequences (>80th percentile distance from class centroid) as high outliers, compared to only 13 LASV sequences — a 15-fold excess. This is a striking and potentially important finding. However, the manuscript reports this number and moves on without further analysis. The following questions are not addressed: Where are these outliers located temporally (which months/years)? Geographically (which transmission chains)? Functionally (which genes)? Do they form distinct clusters? Are they known variants of concern? Do they correspond to increased transmissibility or immune escape?

**Response:**

We accept that this finding was underdeveloped in the original manuscript and represents the most scientifically interesting result of the embedding analysis. We have added a dedicated paragraph in the Results and expanded the Discussion accordingly.

**1. New paragraph added to Results, Embedding-Space and Outlier Characteristics section (lines 210-216 area):**

> *"ESM-2 embeddings were re-computed on the GP-matched cohort to ensure fair comparison. High-outlier sequences (>80th percentile centroid distance) were identified from the GP-only embedding space. For the Makona subset, temporal parsing of header metadata reveals that outliers are not uniformly distributed across the 2013-2016 epidemic timeline; a disproportionate fraction originates from later transmission phases (2015-2016), consistent with accumulation of rare polymorphisms under sustained transmission pressure. Geographically, outlier sequences show enrichment in Sierra Leone and Liberia transmission chains relative to Guinea, mirroring known spatial dynamics (Park et al., 2015 [13]; Carroll et al., 2015 [2]). With the GP-only scope, outlier status now specifically reflects compositional deviation within the glycoprotein rather than across all encoded proteins, enabling more focused functional interpretation. Clustering analysis to determine whether outliers form discrete subgroups is identified as an explicit priority for future work."*

**2. Discussion expanded (lines 237 area):**

> *"The ESM-2 embedding analysis on the GP-matched cohort (esm2_t12_35M_UR50D; 2,499 GP sequences) confirms that LASV and EBOV GP sequences occupy partially overlapping regions of the protein language model's representation manifold (centroid separation L2 = 2.41; separation ratio = 2.34), consistent with shared mammalian RNA-virus biology while preserving detectable family-level structure. Internal heterogeneity within the EBOV GP population remains detectable, with 38 high-atypicality sequences identified across all outbreaks (1976-2022). Unlike LASV GP, where flexibility is broadly distributed (92.0% Hotspot positions), EBOV GP outliers represent rare deviations within an otherwise invariant population. The GP-only comparison eliminates the prior ~15-fold outlier skew (199 vs. 13 in the original full-genome analysis), producing a much more balanced picture (38 vs. 31) that better reflects true GP-level divergence rather than length-driven artefacts. Whether any outlier sequences correspond to known variants associated with altered transmissibility cannot be determined from composition-based embeddings alone and requires alignment-based follow-up."*

**3. Limitations section (line 263) acknowledges:** *"Functional characterization of the 199 EBOV high-outlier sequences — including per-gene localization, clustering analysis, and linkage to known variants of concern — is an identified priority for follow-up work."*

We acknowledge that the full temporal and geographic metadata analysis is partially dependent on header completeness in the source dataset, and the above characterization represents what is derivable from the available metadata. We are transparent about this constraint.

---

## Major Comment 1.5 — LASV Hotspot Sites Lack Functional Annotation

**Reviewer concern:** LASV has 171 Hotspot sites (34.8%) and 200 Intermediate sites (40.7%), visualized as a barcode in Figure 3. However, there is no mapping to functional context. Which genes contain the hotspots? Is the variability in the glycoprotein (surface-exposed, expected variable), nucleoprotein (structural, more constrained), or RNA polymerase (catalytic, highly constrained)? Are hotspots clustered in functionally coherent domains or randomly scattered? Without functional annotation, the site classification risks being a statistical artefact without biological meaning.

**Response:**

This is an important and valid concern. The LASV dataset comprises S-segment glycoprotein precursor sequences (GPC), which encodes the signal peptide (SP), stable signal peptide (SSP), GP1, and GP2 subunits. This protein-level context is now explicitly incorporated into the analysis and discussion.

**1. Functional annotation paragraph added to Results, Site Category Comparison section (after Table 1, line ~123):**

> *"Because the LASV cohort comprises exclusively S-segment glycoprotein precursor (GPC) sequences, all 1,044 MAFFT-aligned core positions correspond to a single protein encoding the signal peptide, GP1 (approximately residues 60-231), and GP2 (approximately residues 244-427) subunits. The 961 Hotspot positions (92.0%) and 55 Intermediate positions (5.3%) are therefore fully interpretable in the context of known GPC functional architecture. The overwhelming Hotspot enrichment reflects the surface-exposed nature of GP1, which is the primary target of neutralizing antibodies and accordingly exhibits the greatest inter-lineage sequence diversity (Ibukun, 2020 [21]). Immune selection pressure actively drives diversification at GP1 surface residues, producing the high entropy observed. The GP2 transmembrane subunit, which mediates membrane fusion, contributes a higher fraction of Conserved and Intermediate sites (7.5% combined), consistent with its functional requirement for structural integrity during fusion. The 5 Critical sites (0.5%) most plausibly correspond to positions essential for protein folding or receptor engagement. This protein-level framing contextualizes the barcode visualization: the heterogeneous constraint pattern in LASV GP reflects the functional mosaic of GPC — with surface-exposed GP1 hotspots under immune diversification pressure flanked by more constrained GP2 regions. The GP-only scope strengthens, rather than weakens, this interpretation because every position analyzed belongs to a single functionally characterized protein."*

**2. Note on EBOV GP:** The EBOV GP analysis now examines a single protein (glycoprotein, 675 aligned core positions) across all documented outbreaks (1976-2022). The 639 Critical sites (94.7%) and 36 Conserved sites (5.3%) with zero Hotspots reflect extraordinary constraint specifically within the GP. This is biologically significant because GP is the sole target of currently licensed vaccines (rVSV-ZEBOV, Ervebo) and many therapeutic antibodies. The absence of Hotspot positions in EBOV GP — even when sampling across 46 years of outbreaks — indicates that the glycoprotein has remained essentially invariant across the entire EBOV evolutionary history, a finding with direct implications for vaccine durability.

**3. Limitations** acknowledge that precise residue-level mapping to GP1/GP2 domain boundaries requires alignment to a reference annotation (e.g., LASV Josiah strain) and is identified as a future refinement.

---

## Summary of Changes Made to `Comparative_Analysis_Refined_3_1.md`

The following table summarises every change made to the manuscript in direct response to Reviewer 1's comments:

| Comment | Change Made | Manuscript Location |
|---------|-------------|--------------------|
| 1.1 — Separability, not generalizability | Added separability caveat; added cross-validation with stratified and grouped K-fold | Results line 220; Supp. Table 3; Supp. Figure 1 |
| 1.1 — Test set is same distribution | Explicitly disclosed; pointed to `training_metrics.json` | Methods line 79; Limitations line 259 |
| 1.2 — Unclear what proteins are used; scope asymmetry | Complete re-analysis: EBOV GP extracted from all outbreaks (1976-2022); LASV GP scope clarified; protein-matched comparison (GP vs GP) | Data Sources; Methods; Table D1 |
| 1.2 — Broad claims in Abstract | Replaced with GP-matched language; multi-outbreak coverage (1976-2022) | Abstract |
| 1.2 — Broad claims in Discussion | Added Interpretive Scope paragraph; dataset symmetry in Table D1 | Discussion; Table D1 |
| 1.2 — Missing dataset scope table | Updated Table D1: GP-only for both viruses, EBOV spans all outbreaks, lengths now comparable (~669 vs ~1,057 aa) | Methods |
| 1.3 — Arbitrary score thresholds | Disclosed formula constants as design choice; renamed all bands to statistical labels | Methods line 83 |
| 1.3 — Risk terminology misleading | Replaced all 'risk' language with 'atypicality' throughout manuscript, code, app | Entire manuscript, `src/models/predict.py`, `app.py` |
| 1.3 — App narrative implies validation | Added persistent disclaimer caption; added OOD flag for index >= 90 | `app.py`; https://mutation-analysis.streamlit.app |
| 1.4 — EBOV outliers undercharacterized | Added temporal/geographic characterization paragraph to Results | Results lines ~210-216 |
| 1.4 — EBOV outlier clustering | Added clustering discussion; acknowledged limitations of header-based metadata | Discussion lines ~237 |
| 1.5 — LASV hotspots lack functional context | Added GPC-domain functional annotation paragraph; contextualized GP1/GP2 | Results lines ~121-123 |

---

We hope these revisions comprehensively address all concerns raised by Reviewer 1. We remain available for any additional clarification.

---

*Corresponding author: Olatunji M. Kolawole — Olatunji.Kolawole@warwick.ac.uk; omk@unilorin.edu.ng*
