"""Generate a publication-quality pipeline flowchart."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(16, 22))
ax.set_xlim(0, 16)
ax.set_ylim(0, 22)
ax.axis('off')

# Color scheme
colors = {
    'input': '#E8F4FD',
    'prep': '#FFF3E0',
    'feat': '#E8F5E9',
    'train': '#F3E5F5',
    'infer': '#E0F7FA',
    'risk': '#FFEBEE',
    'output': '#E8EAF6',
    'border_input': '#1976D2',
    'border_prep': '#F57C00',
    'border_feat': '#388E3C',
    'border_train': '#7B1FA2',
    'border_infer': '#0097A7',
    'border_risk': '#D32F2F',
    'border_output': '#303F9F',
}

def draw_box(ax, x, y, w, h, text, color_key, fontsize=9, bold=False):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.2",
                          facecolor=colors[color_key], edgecolor=colors[f'border_{color_key}'],
                          linewidth=2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            wrap=True, weight=weight, color='#212121')

def draw_arrow(ax, x1, y1, x2, y2, color='#616161'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

# Title
ax.text(8, 21.5, 'Classification & Risk Scoring Pipeline', ha='center', va='center',
        fontsize=18, weight='bold', color='#1565C0')
ax.text(8, 21.0, 'From Raw Sequence to Narrative Interpretation', ha='center', va='center',
        fontsize=11, style='italic', color='#546E7A')

# === STAGE 1: INPUT ===
ax.text(1.5, 20.3, '1. INPUT', fontsize=12, weight='bold', color=colors['border_input'])
draw_box(ax, 1, 19.0, 3, 1.2, 'Raw Sequence\n(FASTA / plain text)', 'input', 10, True)

# === STAGE 2: PREPROCESSING ===
ax.text(5.5, 20.3, '2. PREPROCESSING', fontsize=12, weight='bold', color=colors['border_prep'])
draw_box(ax, 5, 19.0, 4.5, 1.2, 'clean_sequence()\nuppercase + regex filter', 'prep', 9, True)

# Sub-steps for preprocessing
draw_box(ax, 5, 17.8, 1.4, 0.9, 'Whitelist\n20 AA codes', 'prep', 7)
draw_box(ax, 6.6, 17.8, 1.4, 0.9, 'Regex filter\nremove non-AA', 'prep', 7)
draw_box(ax, 8.2, 17.8, 1.3, 0.9, 'Min length\n≥10 residues', 'prep', 7)

# Cleaned sequence
draw_box(ax, 5, 16.5, 4.5, 0.9, 'Cleaned Canonical Sequence', 'prep', 10, True)

# === STAGE 3: FEATURE EXTRACTION ===
ax.text(10.5, 20.3, '3. FEATURE EXTRACTION', fontsize=12, weight='bold', color=colors['border_feat'])
draw_box(ax, 10, 19.0, 5, 1.2, 'amino_acid_frequency_features()\n21-D vector per sequence', 'feat', 9, True)

# Feature details
draw_box(ax, 10, 17.8, 2.4, 0.9, 'seq_length\n(1 feature)', 'feat', 8)
draw_box(ax, 12.6, 17.8, 2.4, 0.9, 'aa_freq_A … aa_freq_Y\n(20 features)', 'feat', 8)

draw_box(ax, 10, 16.5, 5, 0.9, 'Feature Vector: [seq_length, aa_freq_*, …]', 'feat', 9, True)

# === STAGE 4: TRAINING (left branch) ===
ax.text(1.5, 15.5, '4a. TRAINING PHASE', fontsize=12, weight='bold', color=colors['border_train'])
draw_box(ax, 0.5, 14.2, 5.5, 1.1, 'train_best_model()\nstratified 80/20 split | random_state=42', 'train', 8, True)

# Model candidates
draw_box(ax, 0.5, 12.9, 2.7, 1.0, 'LogisticRegression\nmax_iter=300\n+ StandardScaler', 'train', 7)
draw_box(ax, 3.3, 12.9, 2.7, 1.0, 'RandomForest\nn_estimators=300\nclass_weight=balanced', 'train', 7)

# Selection
draw_box(ax, 0.5, 11.7, 5.5, 0.9, 'F1-score selection → best model', 'train', 9, True)

# Persistence
draw_box(ax, 0.5, 10.6, 5.5, 0.8, 'Serialize: joblib.dump() → best_model.joblib', 'train', 8)
draw_box(ax, 0.5, 9.6, 5.5, 0.7, 'metrics.json (accuracy, precision, recall, F1, ROC-AUC)', 'train', 7)

# Risk calibration
draw_box(ax, 0.5, 8.5, 5.5, 0.8, '_build_risk_calibration()', 'train', 9, True)
draw_box(ax, 0.5, 7.5, 2.7, 0.8, 'Per-class centroids\nLassa=0, Ebola=1', 'train', 7)
draw_box(ax, 3.3, 7.5, 2.7, 0.8, 'μ_c = mean distance\nσ_c = std distance', 'train', 7)

# === STAGE 5: INFERENCE (right branch) ===
ax.text(10, 15.5, '4b. INFERENCE PHASE', fontsize=12, weight='bold', color=colors['border_infer'])
draw_box(ax, 9.5, 14.2, 5.5, 1.1, 'predict_sequences()\nload_model_bundle() → best_model.joblib', 'infer', 8, True)

draw_box(ax, 9.5, 12.9, 5.5, 0.9, 'StandardScaler transform\n(using training params)', 'infer', 9)
draw_box(ax, 9.5, 11.8, 5.5, 0.9, 'predict_proba[:, 1] → EBOV probability', 'infer', 9)
draw_box(ax, 9.5, 10.7, 5.5, 0.8, 'Class assignment: threshold ≥ 0.5 → Ebola', 'infer', 9, True)

# === STAGE 6: RISK SCORING ===
ax.text(10, 9.5, '5. RISK SCORING', fontsize=12, weight='bold', color=colors['border_risk'])
draw_box(ax, 9.5, 8.3, 5.5, 1.0, '_compute_risk_scores()', 'risk', 9, True)

# Formula boxes
draw_box(ax, 9.5, 7.0, 5.5, 0.7, 'distance = ||x − centroid_c||₂', 'risk', 9)
draw_box(ax, 9.5, 6.1, 5.5, 0.7, 'z = (distance − μ_c) / σ_c', 'risk', 9)
draw_box(ax, 9.5, 5.2, 5.5, 0.7, 'risk_score = 50.0 + 15.0 × z', 'risk', 9)
draw_box(ax, 9.5, 4.3, 5.5, 0.7, 'clamp(risk_score, 0.0, 100.0)', 'risk', 9)

# Category mapping
ax.text(12.25, 3.7, 'Category Mapping', ha='center', fontsize=9, weight='bold', color=colors['border_risk'])
cats = [('Harmless', '< 20', '#C8E6C9'), ('Neutral', '20–39', '#FFF9C4'), ('Moderate', '40–59', '#FFE0B2'),
        ('Dangerous', '60–79', '#FFCCBC'), ('Critical', '≥ 80', '#EF9A9A')]
for i, (name, rng, clr) in enumerate(cats):
    x = 9.8 + i * 1.1
    box = FancyBboxPatch((x, 2.8), 1.0, 0.7, boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor=clr, edgecolor=colors['border_risk'], linewidth=1)
    ax.add_patch(box)
    ax.text(x+0.5, 3.15, f'{name}\n{rng}', ha='center', va='center', fontsize=6, weight='bold')

# === STAGE 7: OUTPUT ===
ax.text(3, 3.5, '6. OUTPUT', fontsize=12, weight='bold', color=colors['border_output'])

outputs = [
    ('predicted_virus', 'Lassa / Ebola'),
    ('confidence', 'predict_proba value'),
    ('mutation_risk_score', '0 – 100'),
    ('risk_category', '5-tier classification'),
    ('atypicality_zscore', 'distance-based outlier'),
    ('narrative', 'Natural language report'),
]

for i, (label, desc) in enumerate(outputs):
    y = 2.6 - i * 0.45
    draw_box(ax, 0.5, y, 5.5, 0.4, f'{label}\n{desc}', 'output', 7)

# === DEPLOYMENT ===
ax.text(8, 0.8, 'DEPLOYMENT: app.py (Streamlit ≥1.30.0)', ha='center', fontsize=10,
        weight='bold', color='#1565C0')
ax.text(8, 0.4, 'Public URL: https://mutation-analysis.streamlit.app', ha='center', fontsize=9,
        color='#546E7A')
ax.text(8, 0.1, 'Local: streamlit run app.py', ha='center', fontsize=9,
        style='italic', color='#546E7A')

# === ARROWS ===
# Input → Prep
draw_arrow(ax, 2.5, 19.0, 5.0, 19.5)
# Prep main → sub
draw_arrow(ax, 6.5, 19.0, 5.7, 18.7)
draw_arrow(ax, 7.25, 19.0, 7.3, 18.7)
draw_arrow(ax, 8.0, 19.0, 8.9, 18.7)
# Sub → cleaned
draw_arrow(ax, 5.7, 17.8, 6.5, 17.4)
draw_arrow(ax, 7.3, 17.8, 7.25, 17.4)
draw_arrow(ax, 8.9, 17.8, 8.0, 17.4)
# Cleaned → features
draw_arrow(ax, 7.25, 16.5, 9.5, 17.3)
# Prep → cleaned
#draw_arrow(ax, 7.25, 16.5, 7.25, 16.1)

# Features → training/inference
draw_arrow(ax, 12.5, 16.5, 12.75, 15.3)
draw_arrow(ax, 12.5, 15.3, 12.75, 14.2)

# Features split
draw_arrow(ax, 10, 16.5, 3.25, 15.3)
draw_arrow(ax, 3.25, 15.3, 3.25, 14.2)

# Training internals
draw_arrow(ax, 3.25, 14.2, 1.75, 13.9)
draw_arrow(ax, 3.25, 14.2, 4.75, 13.9)
draw_arrow(ax, 1.75, 12.9, 2.5, 12.6)
draw_arrow(ax, 4.75, 12.9, 3.5, 12.6)
draw_arrow(ax, 3.25, 11.7, 3.25, 11.4)
draw_arrow(ax, 3.25, 10.6, 3.25, 10.3)
draw_arrow(ax, 3.25, 9.6, 3.25, 9.3)
draw_arrow(ax, 3.25, 8.5, 1.75, 8.2)
draw_arrow(ax, 3.25, 8.5, 4.75, 8.2)

# Inference internals
draw_arrow(ax, 12.25, 14.2, 12.25, 13.8)
draw_arrow(ax, 12.25, 12.9, 12.25, 12.6)
draw_arrow(ax, 12.25, 11.8, 12.25, 11.5)

# Inference → Risk scoring
draw_arrow(ax, 12.25, 10.7, 12.25, 9.3)

# Risk scoring internals
draw_arrow(ax, 12.25, 8.3, 12.25, 7.7)
draw_arrow(ax, 12.25, 7.0, 12.25, 6.8)
draw_arrow(ax, 12.25, 6.1, 12.25, 5.9)
draw_arrow(ax, 12.25, 5.2, 12.25, 5.0)
draw_arrow(ax, 12.25, 4.3, 12.25, 3.5)

# Training calibration → Risk scoring (data flow)
ax.annotate('', xy=(9.3, 7.9), xytext=(6.0, 7.9),
            arrowprops=dict(arrowstyle='->', color='#7B1FA2', lw=1.5, ls='--'))
ax.text(7.65, 8.15, 'calibration\ndata', ha='center', fontsize=7, color='#7B1FA2', style='italic')

# Risk → outputs
ax.annotate('', xy=(6.0, 2.8), xytext=(9.3, 3.15),
            arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=1.5, ls='--'))

# Class assignment → outputs
ax.annotate('', xy=(6.0, 2.4), xytext=(9.3, 2.4),
            arrowprops=dict(arrowstyle='->', color='#0097A7', lw=1.5, ls='--'))

# Confidence → outputs
ax.annotate('', xy=(6.0, 2.0), xytext=(12.25, 2.0),
            arrowprops=dict(arrowstyle='->', color='#0097A7', lw=1.5, ls='--'))

plt.tight_layout()
plt.savefig('/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/figures/pipeline_flowchart.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/figures/pipeline_flowchart.pdf',
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("Flowchart saved:")
print("  - pipeline_flowchart.png (300 DPI)")
print("  - pipeline_flowchart.pdf (vector)")
