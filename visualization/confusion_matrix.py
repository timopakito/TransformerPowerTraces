import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch.utils.data import DataLoader
from datasets.load_power_traces import PowerTraceDataset, collate_fn
from models.transformer1 import PowerTraceTransformer
from torch.utils.data import random_split
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import seaborn as sns

import matplotlib.pyplot as plt


# 1. Charger le modèle entraîné
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PowerTraceTransformer()
model.load_state_dict(torch.load('weights/best_model.pth', map_location=device))
model = model.to(device)
model.eval()


all_preds = []
all_labels = []
# Charger dataset
dataset = PowerTraceDataset("Data/noisy_data_0.01")

# Split 90/10
n = len(dataset)
n_train = int(0.9 * n)
n_val = n - n_train
train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
val_loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
# 3. Boucle d’évaluation
with torch.no_grad():
    for batch in val_loader:
        inputs, labels, mask = batch  # (B, T, 1), (B, T), (B,)
        inputs, masks, labels = inputs.to(device), mask.to(device), labels.to(device)
        outputs = model(inputs, masks)  # (B, num_classes)
        preds = torch.argmax(outputs, dim=1)  # (B,)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 4. Accuracy globale
acc = accuracy_score(all_labels, all_preds)
print(f"Accuracy: {acc:.4f}")

# 5. Matrice de confusion
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(11), yticklabels=range(11))
plt.xlabel('Prédictions')
plt.ylabel('Véritables')
plt.title('Matrice de confusion avec bruit gaussien, alpha = 0.01')
plt.savefig("visualization/confusion_matrix_little_noisy.png", dpi=300, bbox_inches='tight')
plt.show()
