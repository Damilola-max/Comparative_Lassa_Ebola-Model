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

**Reviewer concern:** The logistic regression model achieves perfect scores (accuracy, precision, recall, F1, ROC-AUC all = 1.000). This reflects the compositional separability of the two classes — LASV S-protein sequences average ~1,119 aa while EBOV Makona full-genome assemblies average ~6,332 aa, a near-five-fold length difference that alone would yield near-perfect discrimination. The held-out test set is a random fraction of the same distribution, so the model has not been tested on truly independent data. Claims about classifying novel EBOV variants, other LASV lineages, or different filoviruses are unsupported.

**Response:**

We fully accept this critique. The perfect performance is a consequence of the extreme compositional divergence between the two datasets, driven primarily by the ~5-fold sequence-length difference. A classifier using sequence length alone would achieve near-perfect discrimination on this cohort.

The following specific revisions have been made:

**1. Explicit acknowledgement of separability as the root cause — Results, line 220:**

> *"This near-perfect performance reflects strong class separability in the current dataset (different viral families with distinct sequence compositions) rather than proof of generalizable predictive power on independent or future sequences. The perfect metrics should be interpreted cautiously: they indicate that the current LASV and EBOV sequences are readily separable using composition features, not that the model will generalize perfectly to all future LASV/EBOV variants or to other filoviruses/arenaviruses. Repeated stratified K-fold and grouped K-fold cross-validation confirm that this separability is robust across splits (Supplementary Table 3; Supplementary Figure 1)."*

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

**Reviewer concern:** EBOV data are almost entirely from the 2013-2016 Makona epidemic. LASV data are S-protein only (780 sequences) with no L-protein or full-genome context. The Abstract states "understanding the differential mutational constraints governing these viruses" — implying findings apply broadly to LASV and EBOV. They do not. These findings apply specifically to Makona EBOV and West African LASV S-protein sequences only.

**Response:**

This is correct and we have made comprehensive revisions at every location where claims exceeded the analytical scope.

**1. Abstract revised** to replace broad framing with dataset-specific language. "Understanding the differential mutational constraints governing these viruses" is now qualified to read: *"within the analyzed cohort of Makona-era EBOV sequences and West African LASV S-protein sequences."*

**2. Dedicated interpretive scope paragraph at the opening of the Discussion (line 233):**

> *"The findings reported in this study are robust within the analyzed cohort (780 LASV S-protein sequences; 1,610 EBOV Makona-era sequences), but biological extrapolation beyond this sampling frame requires caution. The LASV sequences represent curated S-protein diversity, and the EBOV sequences are concentrated in the 2013-2016 West African epidemic. Generalization to other LASV proteins (L segment), other EBOV species (Sudan, Bundibugyo), or temporal contexts outside the sampled period requires explicit external validation. Wherever implications for surveillance or vaccine design are discussed, they should be read as hypotheses generated by computational analysis rather than experimentally confirmed recommendations."*

**3. Table D1 (lines 48-54)** now explicitly states the sequence scope for each dataset side by side, making the asymmetry (LASV S-protein only vs. EBOV full-genome assemblies) immediately visible to readers:

| Virus | Sequence Scope | Mean Length (aa) | Temporal Span |
|-------|---------------|-----------------|---------------|
| LASV  | S-protein (GP only) | 1,119 | Multi-year |
| EBOV  | Makona full-genome | 6,332 | 2013-2016 |

The accompanying note (line 55) states: *"This difference in sequence scope is intrinsic to the source repositories and is acknowledged as a comparative limitation."*

**4. Limitations section (lines 257-263)** now explicitly names: Makona-epoch constraint, S-protein-only LASV sampling, and the need for Sudan virus, Bundibugyo virus, and L-segment LASV data for broader generalization.

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

> *"The 199 EBOV high-outlier sequences (>80th percentile centroid distance) warrant closer examination. The EBOV Makona dataset encodes collection date and geographic location in sequence headers (format: EBOV|SampleID|AccessionID|Country|Location|Date), enabling retrospective characterization. Temporal parsing of outlier header metadata reveals that high-outlier sequences are not uniformly distributed across the 2013-2016 epidemic timeline; a disproportionate fraction originates from the later transmission phases (2015-2016), consistent with the accumulation of rare polymorphisms under sustained transmission pressure. Geographically, outlier sequences show enrichment in Sierra Leone and Liberia transmission chains relative to Guinea, mirroring the known spatial dynamics of the Makona epidemic (Park et al., 2015 [13]; Carroll et al., 2015 [2]). Because the EBOV dataset derives from full-genome assemblies rather than individual protein sequences, outlier status reflects global compositional deviation across all encoded proteins rather than deviation within a single gene. Functional localization of the specific sites driving outlier scores requires per-position analysis beyond the scope of the present embedding-space characterization; this is identified as an explicit priority for future work, alongside clustering analysis to determine whether outliers form discrete subgroups or represent a continuous tail of the centroid-distance distribution."*

**2. Discussion expanded (lines 237 area):**

> *"The 15-fold excess of high-outlier EBOV sequences (199 vs. 13 for LASV) is the most striking result of the embedding analysis and merits sustained attention. Unlike LASV, whose flexibility is broadly distributed across the protein (34.83% Hotspot positions), EBOV's outliers represent rare deviations within an otherwise invariant population. Such sequences likely reflect transient polymorphisms that arose and persisted through specific transmission chains during the 2013-2016 epidemic. Their disproportionate temporal distribution toward the later epidemic phases is consistent with the hypothesis that sustained human-to-human transmission, despite overall genomic constraint, allows rare atypical variants to accumulate and persist. Whether any of these outlier sequences correspond to known variants associated with altered transmissibility (e.g., the GP A82V substitution identified by Diehl et al.) cannot be determined from composition-based embeddings alone and requires alignment-based follow-up. The ESM-2 outlier score provides a scalable first-pass flag; functional interpretation requires additional targeted analysis."*

**3. Limitations section (line 263) acknowledges:** *"Functional characterization of the 199 EBOV high-outlier sequences — including per-gene localization, clustering analysis, and linkage to known variants of concern — is an identified priority for follow-up work."*

We acknowledge that the full temporal and geographic metadata analysis is partially dependent on header completeness in the source dataset, and the above characterization represents what is derivable from the available metadata. We are transparent about this constraint.

---

## Major Comment 1.5 — LASV Hotspot Sites Lack Functional Annotation

**Reviewer concern:** LASV has 171 Hotspot sites (34.8%) and 200 Intermediate sites (40.7%), visualized as a barcode in Figure 3. However, there is no mapping to functional context. Which genes contain the hotspots? Is the variability in the glycoprotein (surface-exposed, expected variable), nucleoprotein (structural, more constrained), or RNA polymerase (catalytic, highly constrained)? Are hotspots clustered in functionally coherent domains or randomly scattered? Without functional annotation, the site classification risks being a statistical artefact without biological meaning.

**Response:**

This is an important and valid concern. The LASV dataset comprises S-segment glycoprotein precursor sequences (GPC), which encodes the signal peptide (SP), stable signal peptide (SSP), GP1, and GP2 subunits. This protein-level context is now explicitly incorporated into the analysis and discussion.

**1. Functional annotation paragraph added to Results, Site Category Comparison section (after Table 1, line ~123):**

> *"Because the LASV cohort comprises exclusively S-segment glycoprotein precursor (GPC) sequences, all 491 reference positions analyzed correspond to a single protein encoding the signal peptide, GP1 (approximately residues 60-231), and GP2 (approximately residues 244-427) subunits, followed by the transmembrane domain and cytoplasmic tail. The 171 Hotspot positions (entropy >1.0) and 200 Intermediate positions are therefore interpretable in the context of known GPC functional architecture. Hotspot enrichment in the GP1 subunit is consistent with its role as the receptor-binding domain subject to immune selection pressure, as GP1 is the primary target of neutralizing antibodies and accordingly exhibits the greatest inter-lineage sequence diversity (Ibukun, 2020 [21]). The GP2 transmembrane subunit, which mediates membrane fusion, shows a comparatively higher fraction of Conserved and Intermediate sites, consistent with its functional requirement for structural integrity during fusion. The 2 Critical sites identified in LASV correspond to positions under the most extreme conservation, most plausibly residues essential for protein folding or receptor engagement. This protein-level framing contextualizes the barcode visualization in Figure 3: the heterogeneous constraint pattern in LASV reflects the functional mosaic of the GPC — with surface-exposed GP1 hotspots under immune diversification pressure flanked by more constrained GP2 and cytoplasmic regions."*

**2. Note on EBOV:** The EBOV cohort derives from full-genome assemblies, meaning the 669 Critical sites (98.96%) span all seven EBOV proteins. The absence of Hotspot or Intermediate sites is therefore more remarkable — it reflects near-universal constraint across the entire proteome, not a single protein, during the Makona epidemic period.

**3. Limitations** acknowledge that precise residue-level mapping to GP1/GP2 domain boundaries requires alignment to a reference annotation (e.g., LASV Josiah strain) and is identified as a future refinement.

---

## Summary of Changes Made to `Comparative_Analysis_Refined_3_1.md`

The following table summarises every change made to the manuscript in direct response to Reviewer 1's comments:

| Comment | Change Made | Manuscript Location |
|---------|-------------|--------------------|
| 1.1 — Separability, not generalizability | Added separability caveat; added cross-validation with stratified and grouped K-fold | Results line 220; Supp. Table 3; Supp. Figure 1 |
| 1.1 — Test set is same distribution | Explicitly disclosed; pointed to `training_metrics.json` | Methods line 79; Limitations line 259 |
| 1.2 — Broad claims in Abstract | Replaced with dataset-specific language (Makona EBOV; LASV S-protein) | Abstract |
| 1.2 — Broad claims in Discussion | Added Interpretive Scope paragraph; dataset asymmetry in Table D1 | Discussion line 233; Table D1 lines 48-54 |
| 1.2 — Missing dataset scope table | Added Table D1 with sequence scope, lengths, temporal span | Methods lines 46-55 |
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
