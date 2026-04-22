# Model README: Lassa vs Ebola Classifier

## What this model does
This model classifies an uploaded protein sequence as either **Lassa** or **Ebola**, then produces an additional **mutation risk score (0-100)** and **risk category**:
- Harmless
- Neutral
- Moderate
- Dangerous
- Critical

The app also generates sentence-level interpretation so non-technical users can understand each result.

## Why this model was chosen
Two candidates were trained and compared on the cleaned dataset:
1. Logistic Regression (with scaling)
2. Random Forest

Both models achieved the same top evaluation metrics on the current split (`accuracy/f1/roc_auc = 1.0`).
The final selected model is **Logistic Regression** because:
- it matched top performance,
- it is lightweight and fast for web inference,
- it is easier to explain and maintain.

## Input data used
- Source file: `data/cleaned/cleaned_sequences.csv`
- Label column: `virus` (`Lassa`, `Ebola`)
- Sequence column: `sequence`

## Feature engineering
The model uses clean sequence-derived numeric features:
- sequence length
- amino-acid frequency profile over standard 20 amino acids

Non-standard characters are removed during preprocessing.

## Risk score method
After prediction, the app computes a risk-style score from an **atypicality distance**:
1. Standardize feature space from training data.
2. Compute class centroids (Lassa/Ebola) in standardized space.
3. Measure each predicted sequence distance to its predicted class centroid.
4. Convert distance to z-score, then to 0-100 risk score.
5. Map score to category bands.

This is not clinical validation; it is a model-based triage signal.

## Saved artifacts
- `models/final/best_model.joblib`: trained model + calibration bundle
- `models/final/training_metrics.json`: training metrics and model selection summary

## How to run
```bash
pip install -r requirements.txt
python3 scripts/03_train.py
streamlit run app.py
```

## App outputs
For each uploaded sequence, the app returns:
- predicted virus
- confidence
- ebola probability
- mutation risk score
- mutation risk category
- atypicality z-score
- plain-language explanation sentence

## Limitations
- This is a computational model, not a clinical or regulatory diagnostic.
- Current feature set is composition-based; it does not use structural biology context.
- Performance depends on distribution of available training sequences.
