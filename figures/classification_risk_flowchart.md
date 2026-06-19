# Classification and Risk Scoring Pipeline Flowchart

```mermaid
flowchart TB
    subgraph INPUT["📥 Input"]
        A["Raw Sequence<br/>(FASTA or plain text)"]
    end

    subgraph PREP["🧹 Preprocessing<br/>src/features/sequence_features.py"]
        B["clean_sequence()<br/>Uppercase + regex filter"]
        B1["Whitelist: A,C,D,E,F,G,H,I,K,L,<br/>M,N,P,Q,R,S,T,V,W,Y"]
        B2["Regex: [^ACDEFGHIKLMNPQRSTVWY]<br/>→ removed"]
        B3["Min length ≥ 10 residues"]
        C["Cleaned Sequence"]
    end

    subgraph FEAT["🔧 Feature Extraction<br/>amino_acid_frequency_features()"]
        D["seq_length"]
        E["aa_freq_A … aa_freq_Y<br/>(20 normalized frequencies)"]
        F["21-D Feature Vector"]
    end

    subgraph TRAIN["🤖 Training Phase<br/>src/models/train.py"]
        G["train_best_model()"]
        H["Stratified 80/20 split<br/>test_size=0.2, random_state=42"]
        I["Candidate Models:"]
        I1["LogisticRegression<br/>max_iter=300 + StandardScaler"]
        I2["RandomForest<br/>n_estimators=300, class_weight='balanced'"]
        J["F1-score selection<br/>Best model → best_model.joblib"]
        K["Risk Calibration:<br/>_build_risk_calibration()"]
        K1["Per-class centroids<br/>(Lassa=0, Ebola=1)"]
        K2["μ_c = mean distance<br/>σ_c = std distance<br/>(clamped ≥ 1e-8)"]
    end

    subgraph INFER["⚡ Inference Phase<br/>src/models/predict.py"]
        L["predict_sequences()"]
        M["StandardScaler transform<br/>(using training params)"]
        N["Model prediction:<br/>predict_proba[:, 1]"]
        O["Class assignment:<br/>threshold ≥ 0.5 → Ebola"]
    end

    subgraph RISK["🎯 Risk Scoring<br/>_compute_risk_scores()"]
        P["distance = ||x − centroid_c||₂"]
        Q["z = (distance − μ_c) / σ_c"]
        R["risk_score = 50.0 + 15.0 × z<br/>clamped [0, 100]"]
    end

    subgraph OUT["📤 Output<br/>app.py (Streamlit)"]
        S["predicted_virus<br/>(Lassa / Ebola)"]
        T["confidence<br/>(predict_proba value)"]
        U["mutation_risk_score<br/>(0-100)"]
        V["Risk Category:<br/>Harmless <20 | Neutral 20-39 |<br/>Moderate 40-59 | Dangerous 60-79 |<br/>Critical ≥80"]
        W["atypicality_zscore"]
        X["Narrative Interpretation<br/>Natural language report"]
    end

    A --> B
    B --> B1
    B1 --> B2
    B2 --> B3
    B3 --> C
    C --> D
    C --> E
    D --> F
    E --> F

    F -->|Training| G
    G --> H
    H --> I
    I --> I1
    I --> I2
    I1 --> J
    I2 --> J
    J --> K
    K --> K1
    K1 --> K2

    F -->|Inference| L
    L --> M
    M --> N
    N --> O
    O --> P
    P --> Q
    Q --> R

    O --> S
    N --> T
    R --> U
    U --> V
    Q --> W
    S --> X
    T --> X
    V --> X
    W --> X
```

## Pipeline Summary

| Stage | File | Key Function | Output |
|-------|------|-------------|--------|
| Preprocessing | `src/features/sequence_features.py` | `clean_sequence()` | Cleaned AA sequence |
| Feature Extraction | `src/features/sequence_features.py` | `amino_acid_frequency_features()` | 21-D vector |
| Training | `src/models/train.py` | `train_best_model()` | `best_model.joblib` |
| Risk Calibration | `src/models/train.py` | `_build_risk_calibration()` | Centroids + μ_c, σ_c |
| Inference | `src/models/predict.py` | `predict_sequences()` | Predictions + risk scores |
| Deployment | `app.py` | Streamlit UI | Interactive web app |

## Risk Score Formula

```
distance = ||x − centroid_c||₂
z = (distance − μ_c) / σ_c
risk_score = clamp(50.0 + 15.0 × z, 0.0, 100.0)
```

## Category Thresholds

| Category | Threshold |
|----------|-----------|
| Harmless | < 20 |
| Neutral | 20 – 39 |
| Moderate | 40 – 59 |
| Dangerous | 60 – 79 |
| Critical | ≥ 80 |
