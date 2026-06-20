import numpy as np
import torch
import pandas as pd
from sklearn.decomposition import PCA

BASE = "/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model"
emb = torch.load(BASE + "/results/gp_revision/gp_embeddings.pt", weights_only=False)
embeddings = np.array(emb["embeddings"])
ids = emb["ids"]

pca = PCA(n_components=2)
pca_coords = pca.fit_transform(embeddings)
var_exp = pca.explained_variance_ratio_ * 100

df_meta = pd.read_csv(BASE + "/data/cleaned/cleaned_sequences_gp_only.csv")
id_to_virus = dict(zip(df_meta["id"].astype(str), df_meta["virus"]))
virus = [id_to_virus.get(str(i), "Unknown") for i in ids]

df = pd.DataFrame({
    "PC1": pca_coords[:, 0],
    "PC2": pca_coords[:, 1],
    "Virus": virus
})
df.to_csv(BASE + "/results/gp_revision/pca_coords.csv", index=False)

with open(BASE + "/results/gp_revision/pca_variance.txt", "w") as f:
    f.write(f"{var_exp[0]:.1f}\n{var_exp[1]:.1f}\n")

print("PCA data saved.")
