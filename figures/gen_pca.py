"""Enhanced PCA figure with density contours and confidence ellipses."""
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.patches import Ellipse
import torch
import pandas as pd
from sklearn.decomposition import PCA

BASE = '/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model'
RESULTS = BASE + '/results/gp_revision'
ASSETS = BASE + '/manuscript/assets/refined3_1/media'

emb = torch.load(RESULTS + '/gp_embeddings.pt', weights_only=False)
embeddings = np.array(emb['embeddings'])
ids = emb['ids']

df_meta = pd.read_csv(BASE + '/data/cleaned/cleaned_sequences_gp_only.csv')
id_to_virus = dict(zip(df_meta['id'].astype(str), df_meta['virus']))
labels = np.array([id_to_virus.get(str(i), 'Unknown') for i in ids])

pca = PCA(n_components=2)
pca_coords = pca.fit_transform(embeddings)

fig, ax = plt.subplots(figsize=(8, 5.5))
virus_colors = {'Ebola': '#c62828', 'Lassa': '#1565c0'}

for virus in ['Ebola', 'Lassa']:
    mask = labels == virus
    ax.scatter(pca_coords[mask, 0], pca_coords[mask, 1],
               c=virus_colors[virus], alpha=0.15, s=12, edgecolors='none', label=virus)

for virus in ['Ebola', 'Lassa']:
    mask = labels == virus
    x = pca_coords[mask, 0]
    y = pca_coords[mask, 1]
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
    ax.contour(Xi, Yi, zi, levels=3, colors=virus_colors[virus], linewidths=1.5, alpha=0.7)

for virus in ['Ebola', 'Lassa']:
    mask = labels == virus
    x = pca_coords[mask, 0]
    y = pca_coords[mask, 1]
    cov = np.cov(x, y)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                  width=lambda_[0]*2*np.sqrt(5.991),
                  height=lambda_[1]*2*np.sqrt(5.991),
                  angle=np.rad2deg(np.arccos(v[0, 0])),
                  edgecolor=virus_colors[virus], facecolor='none',
                  linewidth=2.5, linestyle='--', alpha=0.9)
    ax.add_patch(ell)
    ax.scatter(np.mean(x), np.mean(y), c=virus_colors[virus], s=150,
               marker='X', edgecolors='white', linewidths=1.5, zorder=5)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
ax.set_title('ESM-2 Embedding PCA — GP Sequences (n=2,499)', fontsize=13, fontweight='bold')
ax.legend(title='Virus', loc='upper right', frameon=True, fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(ASSETS + '/image6.png', dpi=300, bbox_inches='tight')
fig.savefig(ASSETS + '/image6.svg', bbox_inches='tight')
plt.close(fig)
print('Enhanced PCA saved.')
