"""Generate a vertical pipeline flowchart for Figure 1."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(8, 14))
ax.set_xlim(0, 8)
ax.set_ylim(0, 14)
ax.axis('off')

# Color scheme
colors = {
    'input': '#E3F2FD',
    'prep': '#FFF3E0',
    'feat': '#E8F5E9',
    'train': '#F3E5F5',
    'infer': '#E0F7FA',
    'risk': '#FFEBEE',
    'output': '#E8EAF6',
    'border_input': '#1565C0',
    'border_prep': '#EF6C00',
    'border_feat': '#2E7D32',
    'border_train': '#6A1B9A',
    'border_infer': '#00838F',
    'border_risk': '#C62828',
    'border_output': '#283593',
}

def draw_box(ax, x, y, w, h, text, color_key, fontsize=9, bold=False):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.15",
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
ax.text(4, 13.6, 'ESM-embedR Pipeline', ha='center', va='center',
        fontsize=16, weight='bold', color='#1565C0')
ax.text(4, 13.25, 'From raw sequence to narrative interpretation', ha='center', va='center',
        fontsize=10, style='italic', color='#546E7A')

# Stage Y positions (top to bottom)
stages = [
    ('1. INPUT', 'input', 'border_input', 12.4, [
        ('Raw sequence\n(FASTA / plain text)', 'input', 11.6, 2.5, 0.7),
    ]),
    ('2. PREPROCESSING', 'prep', 'border_prep', 10.8, [
        ('clean_sequence()\nuppercase + regex filter', 'prep', 10.0, 4.5, 0.7),
        ('Whitelist 20 AA codes', 'prep', 9.2, 2.1, 0.55),
        ('Regex filter\nnon-AA removal', 'prep', 9.2, 2.1, 0.55, 2.4),
        ('Min length ≥10', 'prep', 9.2, 2.1, 0.55, 4.8),
        ('Cleaned canonical sequence', 'prep', 8.4, 4.5, 0.6),
    ]),
    ('3. FEATURE EXTRACTION', 'feat', 'border_feat', 7.6, [
        ('amino_acid_frequency_features()\n21-D vector per sequence', 'feat', 6.8, 4.5, 0.7),
        ('seq_length (1)', 'feat', 6.0, 2.1, 0.55),
        ('aa_freq_A … Y (20)', 'feat', 6.0, 2.1, 0.55, 2.4),
        ('Feature vector [seq_length, aa_freq_*]', 'feat', 5.2, 4.5, 0.6),
    ]),
    ('4. TRAINING', 'train', 'border_train', 4.5, [
        ('train_best_model()\nstratified 80/20 split', 'train', 3.7, 4.5, 0.7),
        ('LogisticRegression\n+ StandardScaler', 'train', 2.9, 2.1, 0.6),
        ('RandomForest\nclass_weight=balanced', 'train', 2.9, 2.1, 0.6, 2.4),
        ('F1 selection → best model', 'train', 2.1, 4.5, 0.55),
        ('Serialize: joblib.dump()', 'train', 1.4, 4.5, 0.5),
        ('_build_risk_calibration()', 'train', 0.7, 4.5, 0.55),
    ]),
]

for title, _, border_color, title_y, boxes in stages:
    ax.text(4, title_y, title, fontsize=11, weight='bold', color=colors[border_color], ha='center')
    prev_y = None
    prev_x = 4
    for box in boxes:
        text, key, y, w, h = box[0], box[1], box[2], box[3], box[4]
        x = box[5] if len(box) > 5 else (8 - w) / 2
        draw_box(ax, x, y, w, h, text, key, fontsize=8, bold=(h >= 0.6))
        if prev_y is not None:
            draw_arrow(ax, prev_x, prev_y, x + w/2, y + h)
        prev_y = y
        prev_x = x + w/2

# Arrows between stages
draw_arrow(ax, 4, 11.6, 4, 11.0)
draw_arrow(ax, 4, 10.0, 4, 9.75)
draw_arrow(ax, 4, 8.4, 4, 7.95)
draw_arrow(ax, 4, 5.2, 4, 4.95)

# Inference branch (side column on right)
ax.text(6.5, 4.5, '5. INFERENCE', fontsize=10, weight='bold', color=colors['border_infer'], ha='center')
inf_boxes = [
    ('predict_sequences()\nload model', 'infer', 3.7, 2.2, 0.6),
    ('StandardScaler\n(training params)', 'infer', 3.0, 2.2, 0.55),
    ('predict_proba → EBOV prob', 'infer', 2.3, 2.2, 0.55),
    ('Class assignment\nthreshold ≥ 0.5', 'infer', 1.6, 2.2, 0.55),
]
prev_y = None
for text, key, y, w, h in inf_boxes:
    x = 6.5 - w/2
    draw_box(ax, x, y, w, h, text, key, fontsize=7, bold=(h >= 0.6))
    if prev_y is not None:
        draw_arrow(ax, 6.5, prev_y, 6.5, y + h)
    prev_y = y

# Risk scoring (side column on right, below inference)
ax.text(6.5, 1.3, '6. RISK SCORING', fontsize=10, weight='bold', color=colors['border_risk'], ha='center')
risk_boxes = [
    ('_compute_risk_scores()', 'risk', 0.6, 2.2, 0.5),
    ('z = (dist − μ_c) / σ_c', 'risk', 0.0, 2.2, 0.45),
]
prev_y = None
for text, key, y, w, h in risk_boxes:
    x = 6.5 - w/2
    draw_box(ax, x, y, w, h, text, key, fontsize=7, bold=(h >= 0.5))
    if prev_y is not None:
        draw_arrow(ax, 6.5, prev_y, 6.5, y + h)
    prev_y = y

# Arrow from training calibration to risk scoring
ax.annotate('', xy=(5.4, 0.85), xytext=(2.75, 0.85),
            arrowprops=dict(arrowstyle='->', color='#7B1FA2', lw=1.5, ls='--'))
ax.text(4.05, 0.95, 'calibration data', ha='center', fontsize=6, color='#7B1FA2', style='italic')

# Arrow from inference to risk scoring
ax.annotate('', xy=(5.4, 1.6), xytext=(6.5, 1.6),
            arrowprops=dict(arrowstyle='->', color='#616161', lw=1.2))

# Output summary at bottom left
ax.text(1.5, 0.3, '7. OUTPUT', fontsize=10, weight='bold', color=colors['border_output'], ha='center')
output_text = 'predicted_virus  |  confidence  |  risk_score  |  narrative'
ax.text(1.5, 0.05, output_text, fontsize=7, ha='center', color='#37474F', family='monospace')

# Arrow from risk to output
ax.annotate('', xy=(2.4, 0.2), xytext=(5.4, 0.2),
            arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=1.5, ls='--'))

# Deployment note
ax.text(4, -0.35, 'Deployment: app.py (Streamlit)  |  https://mutation-analysis.streamlit.app',
        ha='center', fontsize=8, weight='bold', color='#1565C0')

plt.tight_layout()
plt.savefig('/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/manuscript/assets/refined3_1/media/image1.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/manuscript/assets/refined3_1/media/image1.svg',
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("Vertical flowchart saved.")
