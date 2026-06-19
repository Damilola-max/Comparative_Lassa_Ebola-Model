"""Generate ESM-2 (35M) embeddings for GP-only sequences with checkpoint/resume."""
import torch
import esm
import pandas as pd
from tqdm import tqdm
import os
import time

base = "/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model"
checkpoint_path = f"{base}/results/gp_revision/gp_embeddings_checkpoint.pt"
final_path = f"{base}/results/gp_revision/gp_embeddings.pt"

# Load data
df = pd.read_csv(f"{base}/data/cleaned/cleaned_sequences_gp_only.csv")
df = df[df["length"] >= 300].copy()

MODEL_NAME = "esm2_t12_35M_UR50D"
print(f"Loading ESM-2 model ({MODEL_NAME})...")
model, alphabet = esm.pretrained.load_model_and_alphabet(MODEL_NAME)
model = model.eval()
print(f"Model loaded. Using CPU.")

batch_converter = alphabet.get_batch_converter()

# Prepare sequences
sequences = [(row["id"], row["sequence"]) for _, row in df.iterrows()]
total = len(sequences)
print(f"Total sequences to embed: {total}")

# Resume from checkpoint if exists AND matches model
start_idx = 0
all_embeddings = []
all_ids = []
if os.path.exists(checkpoint_path):
    print(f"\n=== CHECKPOINT FOUND ===")
    ckpt = torch.load(checkpoint_path, weights_only=False)
    if ckpt.get("model") == MODEL_NAME:
        start_idx = ckpt["next_idx"]
        all_embeddings = ckpt["embeddings"]
        all_ids = ckpt["ids"]
        print(f"Resuming: {len(all_ids)} done, {total - len(all_ids)} remaining")
    else:
        print(f"Checkpoint uses different model ({ckpt.get('model', 'unknown')}), starting fresh.")
        os.remove(checkpoint_path)
else:
    print(f"\n=== STARTING FRESH ===")

batch_size = 16
checkpoint_every = 10  # Save every 10 batches

start_time = time.time()
for i in range(start_idx, total, batch_size):
    batch = sequences[i:i+batch_size]
    
    try:
        _, _, tokens = batch_converter(batch)
        with torch.no_grad():
            results = model(tokens, repr_layers=[12], return_contacts=False)
        
        for j, (seq_id, seq) in enumerate(batch):
            emb = results["representations"][12][j, 1:len(seq)+1].mean(0).cpu()
            all_embeddings.append(emb)
            all_ids.append(seq_id)
    except Exception as e:
        print(f"\nERROR at index {i}: {e}")
        torch.save({"model": MODEL_NAME, "next_idx": i, "embeddings": all_embeddings, "ids": all_ids}, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        raise
    
    batch_num = (i // batch_size) + 1
    if batch_num % checkpoint_every == 0:
        torch.save({"model": MODEL_NAME, "next_idx": i + batch_size, "embeddings": all_embeddings, "ids": all_ids}, checkpoint_path)
        elapsed = time.time() - start_time
        rate = len(all_embeddings) / elapsed if elapsed > 0 else 0
        pct = len(all_embeddings) / total * 100
        print(f"  [Checkpoint] {len(all_embeddings)}/{total} ({pct:.1f}%) | Batch {batch_num}/{(total-1)//batch_size + 1} | Rate: {rate:.2f} seq/s | Elapsed: {elapsed/60:.1f}min")

# Final save
print(f"\n=== FINALIZING ===")
emb_tensor = torch.stack(all_embeddings)
torch.save({"ids": all_ids, "embeddings": emb_tensor, "model": MODEL_NAME}, final_path)
print(f"Final embeddings saved: {final_path}")
print(f"Shape: {emb_tensor.shape}")

if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)
    print("Checkpoint removed.")

print("\n=== DONE ===")
