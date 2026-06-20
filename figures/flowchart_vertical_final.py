import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(8, 14))
ax.set_xlim(0, 8)
ax.set_ylim(0, 14)
ax.axis('off')

C_BR = '#37474F'
C_BL = '#1565C0'
C_T = '#263238'
C_HL = '#E3F2FD'
C_LG = '#F5F5F5'

def box(cx, cy, w, h, txt, hl=False, fs=10):
    fc, ec, lw = (C_HL, C_BL, 2.5) if hl else (C_LG, C_BR, 1.5)
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                boxstyle="round,pad=0.03", fc=fc, ec=ec, lw=lw))
    ax.text(cx, cy, txt, ha='center', va='center', fontsize=fs, color=C_T,
            weight='bold' if hl else 'normal', linespacing=1.25)

def arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color='#546E7A', lw=2, mutation_scale=15))

# Pure vertical flow — single column, no sideways branching, no title/footer
steps = [
    (12.5, "Raw Sequence\nFASTA or plain text", True),
    (11.3, "clean_sequence()\nuppercase + regex filter", False),
    (10.1, "Cleaned Canonical Sequence", True),
    ( 8.9, "Feature Extraction\n21-D vector per sequence", True),
    ( 7.7, "Feature Vector\n[seq_length, aa_freq_A … Y]", False),
    ( 6.5, "Model Training\nLogisticRegression  |  RandomForest", True),
    ( 5.3, "F1 Selection → Best Model", True),
    ( 4.1, "Inference\npredict_proba + class assignment", False),
    ( 2.9, "Risk Scoring\n_compute_risk_scores()", True),
    ( 1.7, "Output\npredicted_virus | confidence | risk_score | narrative", False),
]

W = 5.0
H = 0.95
for i, (y, txt, hl) in enumerate(steps):
    box(4, y, W, H, txt, hl, 9.5)
    if i < len(steps) - 1:
        arrow(4, y - H/2, 4, steps[i+1][0] + H/2)

plt.tight_layout()
plt.savefig('/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/manuscript/assets/refined3_1/media/image1.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/manuscript/assets/refined3_1/media/image1.svg',
            bbox_inches='tight', facecolor='white')
print('Done: pure vertical flowchart')
