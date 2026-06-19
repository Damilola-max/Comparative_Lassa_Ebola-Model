import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

fig, ax = plt.subplots(figsize=(12, 16))
ax.set_xlim(0, 12)
ax.set_ylim(0, 16)
ax.axis('off')

C_BOX = '#F5F5F5'
C_HL = '#E3F2FD'
C_ACC = '#FFF3E0'
C_BR = '#37474F'
C_T = '#263238'
C_BL = '#1565C0'

def box(x, y, w, h, txt, hl=False, fs=11):
    fc, ec, lw = (C_HL, C_BL, 2.5) if hl else (C_BOX, C_BR, 1.5)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=fc, ec=ec, lw=lw))
    ax.text(x + w/2, y + h/2, txt, ha='center', va='center', fontsize=fs, color=C_T, weight='bold' if hl else 'normal', linespacing=1.3)

def arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='-|>', color='#546E7A', lw=2, mutation_scale=15))

# Title
ax.text(6, 15.5, 'Classification & Risk Scoring Pipeline', ha='center', fontsize=18, weight='bold', color=C_BL)

# 3-column layout
# Col 1: Preprocessing
ax.text(2, 14.5, 'STEP 1: Preprocessing', ha='center', fontsize=12, weight='bold', color=C_BL)
box(0.2, 13.2, 3.6, 1.0, 'Raw Sequence\nFASTA / plain text', hl=True, fs=10)
arrow(2, 13.2, 2, 12.5)
box(0.2, 11.5, 3.6, 0.9, 'clean_sequence()\nRegex filter, min length ≥ 10', fs=9)
arrow(2, 11.5, 2, 10.8)
box(0.2, 9.8, 3.6, 0.9, 'Cleaned Sequence\n20 canonical amino acids', hl=True, fs=10)

# Col 2: Features
ax.text(6, 14.5, 'STEP 2: Features', ha='center', fontsize=12, weight='bold', color=C_BL)
box(4.2, 13.2, 3.6, 1.0, 'Feature Extraction\n21-D vector per sequence', hl=True, fs=10)
arrow(6, 13.2, 6, 12.5)
box(4.2, 11.5, 1.7, 0.9, 'seq_length\n(1)', fs=9)
box(6.1, 11.5, 1.7, 0.9, 'aa_freq_A…Y\n(20)', fs=9)
arrow(5.05, 11.5, 5.5, 10.8)
arrow(6.95, 11.5, 6.5, 10.8)
box(4.2, 9.8, 3.6, 0.9, 'Feature Vector\n[seq_length, aa_freq_*]', hl=True, fs=10)

# Col 3: Training
ax.text(10, 14.5, 'STEP 3: Training', ha='center', fontsize=12, weight='bold', color=C_BL)
box(8.2, 13.2, 3.6, 1.0, '80/20 Split\nrandom_state = 42', fs=10)
arrow(10, 13.2, 10, 12.5)
box(8.2, 11.5, 1.7, 0.9, 'Logistic\nRegression', fs=9)
box(10.1, 11.5, 1.7, 0.9, 'Random\nForest', fs=9)
arrow(9.05, 11.5, 9.5, 10.8)
arrow(10.95, 11.5, 10.5, 10.8)
box(8.2, 9.8, 3.6, 0.9, 'F1 Selection\n→ best_model.joblib', hl=True, fs=10)

# Horizontal arrows between columns
arrow(3.8, 10.35, 4.2, 10.35)
arrow(7.8, 10.35, 8.2, 10.35)

# Bottom row: Inference & Risk
ax.text(2, 8.5, 'Calibration', ha='center', fontsize=11, weight='bold', color='#E65100')
ax.text(6, 8.5, 'Inference', ha='center', fontsize=11, weight='bold', color='#E65100')
ax.text(10, 8.5, 'Risk Scoring', ha='center', fontsize=11, weight='bold', color='#E65100')

box(0.2, 7.0, 3.6, 1.2, 'Per-class centroids\nμ_c = mean, σ_c = std\n(σ_c ≥ 1e-8)', fs=9)
box(4.2, 7.0, 3.6, 1.2, 'StandardScaler\npredict_proba[:, 1]\nthreshold ≥ 0.5', fs=9)
box(8.2, 7.0, 3.6, 1.2, '||x − centroid_c||₂\nz = (dist − μ_c) / σ_c\nrisk = 50 + 15 × z', hl=True, fs=9)

arrow(2, 7.0, 2, 6.3)
arrow(6, 7.0, 6, 6.3)
arrow(10, 7.0, 10, 6.3)

# Output row
ax.text(6, 5.7, 'OUTPUT', ha='center', fontsize=13, weight='bold', color=C_BL)
box(0.2, 4.2, 2.2, 1.3, 'Predicted\nVirus', hl=True, fs=11)
box(2.6, 4.2, 2.2, 1.3, 'Confidence\n(probability)', hl=True, fs=11)
box(5.0, 4.2, 2.2, 1.3, 'Risk\nScore', hl=True, fs=11)
box(7.4, 4.2, 2.2, 1.3, 'Risk\nCategory', hl=True, fs=11)
box(9.8, 4.2, 2.0, 1.3, 'Z-score\n(atypicality)', hl=True, fs=11)

# Category thresholds
ax.text(6, 3.6, 'Category Thresholds', ha='center', fontsize=10, weight='bold', color=C_T)
cats = [('Harmless', '<20', '#C8E6C9'), ('Neutral', '20-39', '#FFF9C4'), ('Moderate', '40-59', '#FFE0B2'),
        ('Dangerous', '60-79', '#FFCCBC'), ('Critical', '≥80', '#EF9A9A')]
for i, (n, v, c) in enumerate(cats):
    x = 0.8 + i * 2.15
    ax.add_patch(Rectangle((x, 2.7), 1.9, 0.7, facecolor=c, edgecolor=C_BR, linewidth=1))
    ax.text(x + 0.95, 3.05, f'{n}\n{v}', ha='center', va='center', fontsize=8, weight='bold', color=C_T)

# Deployment footer
ax.text(6, 2.2, 'Deployment: Streamlit app', ha='center', fontsize=10, weight='bold', color=C_BL)
ax.text(6, 1.8, 'https://mutation-analysis.streamlit.app  |  streamlit run app.py', ha='center', fontsize=9, color='#546E7A')

plt.tight_layout()
plt.savefig('/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/figures/pipeline_flowchart_clean.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/figures/pipeline_flowchart_clean.pdf', bbox_inches='tight', facecolor='white')
print('Done: pipeline_flowchart_clean.png and .pdf')
