import torch
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

from datasets.load_power_traces import PowerTraceDataset, collate_fn_val
from models.transformer1 import PowerTraceTransformer

# === Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
MODEL_PATH = "weights/best_model_bis.pth"
DATA_DIR = "Data"

# === Charger le dataset complet et créer un subset validation cohérent
full_dataset = PowerTraceDataset(DATA_DIR)
labels = full_dataset.labels

# Charger l'index de split (reproduire celui utilisé à l'entraînement)
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
_, val_idx = next(sss.split(X=range(len(labels)), y=labels))
val_subset = Subset(full_dataset, val_idx)

val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_val)

# === Charger le modèle
model = PowerTraceTransformer(num_classes=len(full_dataset.label_map)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# === Ajouter méthode d'extraction des embeddings (si pas déjà dans le modèle)
def extract_embedding(self, x, mask):
    self.eval()
    with torch.no_grad():
        x = self.input_proj(x)  # (B, T, d_model)
        B, T, _ = x.size()
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)  # (B, T, d_model)

        padding_mask = ~mask  # (B, T) — True = à ignorer
        x = self.encoder(x, src_key_padding_mask=padding_mask)  # (B, T, d_model)

        # On renvoie l'embedding moyen (ou CLS pooling si tu préfères)
        return x.mean(dim=1)  # (B, d_model)
model.extract_embedding = extract_embedding.__get__(model)

# === Extraire les embeddings
print("🔍 Extraction des embeddings du jeu de validation...")

embeddings = []
labels = []

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)
        y = torch.tensor([y]).to(device)

        L = x.shape[0]
        window_size = L
        mask = torch.ones(1, window_size, dtype=torch.bool).to(device)

        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(-1)  # (T,) → (1, T, 1)
        elif x.ndim == 2:
            x = x.unsqueeze(0)  # (T, 1) → (1, T, 1)
        # Sinon, x est déjà de forme (1, T, 1) → pas besoin de modifier
        mask = torch.ones(x.shape[1], dtype=torch.bool).unsqueeze(0).to(device)

        emb = model.extract_embedding(x, mask)  # (1, D)

        embeddings.append(emb.cpu())
        labels.extend(y.cpu().numpy())


embeddings = torch.cat(embeddings, dim=0).numpy()
labels = np.array(labels)

# === Réduction de dimension et t-SNE
print("📉 Réduction des dimensions (PCA + t-SNE)...")
pca = PCA(n_components=50).fit_transform(embeddings)
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
proj = tsne.fit_transform(pca)

# === Visualisation
print("📊 Affichage de la projection t-SNE...")
plt.figure(figsize=(10, 8))
scatter = plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab10", alpha=0.7)
plt.legend(*scatter.legend_elements(), title="Classe")
plt.title("Visualisation t-SNE des embeddings du Transformer")
plt.grid(True)
plt.savefig("tsne_embeddings.png")
plt.show()
