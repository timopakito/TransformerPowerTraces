from collections import defaultdict
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from datasets.load_power_traces import PowerTraceDatasetSlidingWindow, collate_fn_unified,SlidingWindowWrapper, collate_fn_val, PaddedShortTraceDataset, BalancedSegmentDataset
from datasets.load_power_traces import PowerTraceDataset, collate_fn
from models.transformer1 import PowerTraceTransformer
from torch.utils.data import random_split
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
from analyze_attention import plot_attention_distribution

# === Configuration du device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Charger les datasets
full_dataset = PowerTraceDataset("Data")


# === Split 80/20 sur le dataset complet ===
n = len(full_dataset)
n_train = int(0.8 * n)
n_val = n - n_train

# Liste des labels
labels = full_dataset.labels  # ← déjà dispo dans PowerTraceDataset
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
train_idx, val_idx = next(sss.split(X=range(len(labels)), y=labels))

train_subset = Subset(full_dataset, train_idx)
val_subset = Subset(full_dataset, val_idx)

# === Création des DataLoaders, appliquer le sliding window sur le dataset d'entrainement ===
train_dataset_windowed = SlidingWindowWrapper(train_subset, window_size=54, step_size=27)
#train_dataset_short = PaddedShortTraceDataset(train_subset)
# Choisis le nombre de segments cibles par classe
segments_per_class = 80  # Par exemple

# Crée le dataset équilibré
balanced_dataset = BalancedSegmentDataset(
    datasets=[train_dataset_windowed],
    segments_per_class=segments_per_class
)
val_dataset = val_subset  # Restent sous forme brute
#Loader
train_loader = DataLoader(balanced_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_unified)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_val)






# === Créer le modèle ===
model = PowerTraceTransformer(num_classes=len(full_dataset.label_map)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

# === Liste pour stocker les poids d'attention ===
attention_data = []

# === Fonction de validation avec perte et prédiction agrégée ===
def validate(model, val_loader, criterion, device, label_map=full_dataset.label_map, window_size=54, step_size=27):
    model.eval()
    total = correct = 0
    val_loss = 0
    all_preds = []
    all_labels = []
    attention_data = []
    skipped = 0
    class_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    error_examples = []

    with torch.no_grad():
        for idx, (full_trace, label) in enumerate(val_loader):
            full_trace = full_trace.to(device)
            label = torch.tensor(label).to(device)

            segment_preds = []
            segment_labels = []
            weights_all = []

            L = full_trace.size(0)
            if full_trace.ndim == 2:
                full_trace = full_trace.squeeze(-1)

            if L < window_size:
                padded = torch.zeros(window_size, device=device)
                padded[:L] = full_trace
                segment = padded.unsqueeze(0).unsqueeze(-1)
                mask = torch.zeros(window_size, dtype=torch.bool, device=device)
                mask[:L] = True
                mask = mask.unsqueeze(0)

                out = model(segment, mask)
                loss = criterion(out, label.unsqueeze(0))
                val_loss += loss.item()

                pred = out.argmax(dim=1).item()
                segment_preds.append(pred)
                segment_labels.append(label.item())
                correct += int(pred == label.item())
                total += 1

                class_stats[label.item()]["total"] += 1
                if pred == label.item():
                    class_stats[label.item()]["correct"] += 1
                else:
                    error_examples.append((full_trace[:L].cpu().numpy(), pred, label.item()))

                try:
                    weights = model.get_attention_weights()
                    weights_all.append(weights[0].detach().cpu().numpy())
                except:
                    weights_all.append([[0.0] * window_size])
            else:
                for start in range(0, L - window_size + 1, step_size):
                    segment = full_trace[start:start + window_size].unsqueeze(0).unsqueeze(-1).to(device)
                    mask = torch.ones(1, window_size, dtype=torch.bool).to(device)

                    out = model(segment, mask)
                    loss = criterion(out, label.unsqueeze(0))
                    val_loss += loss.item()

                    pred = out.argmax(dim=1).item()
                    segment_preds.append(pred)
                    segment_labels.append(label.item())
                    correct += int(pred == label.item())
                    total += 1

                    class_stats[label.item()]["total"] += 1
                    if pred == label.item():
                        class_stats[label.item()]["correct"] += 1
                    else:
                        error_examples.append((segment.squeeze(-1).squeeze(0).cpu().numpy(), pred, label.item()))

                    try:
                        weights = model.get_attention_weights()
                        weights_all.append(weights[0].detach().cpu().numpy())
                    except:
                        weights_all.append([[0.0] * window_size])

            if len(segment_preds) == 0:
                skipped += 1
                continue

            all_preds.extend(segment_preds)
            all_labels.extend(segment_labels)

            attention_data.append({
                "trace_id": idx,
                "label": label.item(),
                "attention_weights": weights_all
            })

    if total == 0:
        print(f"⚠️ Aucune trace valide en validation. {skipped} traces ont été ignorées (trop courtes).")
        return 0.0, float('inf'), [], [], []

    acc = correct / total
    avg_loss = val_loss / total
    print(f"\n✅ Accuracy globale (segments) : {acc:.4f}")
    print("🧾 Répartition des classes (segments):")
    print(np.bincount(all_labels, minlength=len(label_map)))

    # — Accuracy par classe
    print("\n📊 Accuracy par classe (segments) :")
    for class_id in range(len(label_map)):
        stats = class_stats[class_id]
        total_c = stats["total"]
        correct_c = stats["correct"]
        if total_c > 0:
            acc_c = correct_c / total_c
            print(f"  Classe {label_map[class_id]} ({class_id}) : {acc_c:.2%} ({correct_c}/{total_c})")
        else:
            print(f"  Classe {label_map[class_id]} ({class_id}) : aucune donnée")
    return acc, avg_loss, all_preds, all_labels, attention_data

#Entrainement 
best_acc = 0.0
train_losses, val_losses = [], []

for epoch in range(10):
    model.train()
    total_loss = 0  

    for x, y, mask in train_loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device).bool()
        out = model(x, mask)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_losses.append(total_loss)
    print(f"📚 Epoch {epoch+1} - Train Loss: {total_loss:.4f}")

    # === Validation
    val_acc, val_loss_epoch, all_preds, all_labels, attention_data = validate(
        model, val_loader, criterion, device, window_size=54, step_size=27
    )
    val_losses.append(val_loss_epoch)

    print(f"✅ Epoch {epoch+1} - Val Accuracy: {val_acc:.2%} - Val Loss: {val_loss_epoch:.4f}")

    # === Sauvegarde du meilleur modèle
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "weights/best_model_bis.pth")
        print(f"💾 Nouveau meilleur modèle sauvegardé avec {best_acc:.2%} de précision !")


# === Sauvegarde des poids d'attention ===
df_attention = pd.DataFrame(attention_data)


# Fonction pour convertir et valider les poids d'attention
def safe_json_dumps(val):
    try:
        # Log the raw value for debugging
        #print(f"Serializing value: {val} (type: {type(val)})")

        # Convertir en liste Python
        if isinstance(val, (np.ndarray, torch.Tensor)):
            val = val.tolist()  # Convertir numpy array ou tenseur en liste
        elif isinstance(val, (list, tuple)):
            val = list(val)  # Convertir tuple en liste
        else:
            raise ValueError(f"Invalid type for attention weights: {type(val)}")

        # Si val est une liste vide, la gÃ©rer explicitement
        if not val:
            raise ValueError("Attention weights list is empty")

        # Convertir rÃ©cursivement les arrays imbriquÃ©s en listes
        def convert_nested_arrays(obj):
            if isinstance(obj, (np.ndarray, torch.Tensor)):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_nested_arrays(item) for item in obj]
            elif isinstance(obj, (int, float)):
                return obj
            else:
                raise ValueError(f"Invalid type in attention weights: {type(obj)}")

        val = convert_nested_arrays(val)

        # VÃ©rifier que c'est une liste de listes de nombres
        if not all(isinstance(sublist, list) for sublist in val):
            #print(f"Converting flat list to list of lists: {val}")
            val = [val]  # Envelopper une liste plate, si nÃ©cessaire

        if not all(all(isinstance(x, (int, float)) for x in sublist) for sublist in val):
            raise ValueError("Attention weights must contain only numbers")

        # SÃ©rialiser en JSON
        serialized = json.dumps(val)
        #print(f"Serialized successfully: {serialized}")
        return serialized

    except Exception as e:
        print(f"âš ï¸ Erreur lors de la sÃ©rialisation des poids d'attention: {str(e)} (value: {val})")
        return "[]"  # Retourner une liste vide seulement en cas d'erreur explicite

# Appliquer la conversion et sÃ©rialisation
df_attention["attention_weights"] = df_attention["attention_weights"].apply(safe_json_dumps)

'''# Vérifier le contenu après sérialisation
print("df_attention after serialization:")
print(df_attention["attention_weights"])'''

# Sauvegarder dans le CSV
df_attention.to_csv("attention_weights.csv", index=False)
print("✅ Poids d'attention sauvegardés dans attention_weights.csv")

# === Rapport de classification ===
print("\n\n=== Rapport de classification ===")
print(classification_report(
    all_labels, 
    all_preds, 
    target_names=list(full_dataset.label_map.values()),
    labels=list(range(len(full_dataset.label_map))) 
))

# === Matrice de confusion ===
print("\n=== Matrice de confusion ===")
cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(full_dataset.label_map))))
ConfusionMatrixDisplay(cm, display_labels=list(full_dataset.label_map.values())).plot()
plt.title("Confusion matrix - Validation")
plt.savefig("confusion_matrix.png")
plt.show()

# === Courbe de pertes ===
plt.figure()
plt.plot(train_losses, label="Train loss")
plt.plot(val_losses, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Courbe de perte")
plt.savefig("loss_graph.png")
plt.grid(True)
plt.show()


# === Visualiser les traces avec les poids d'attention et les sauvegarder ===
print("📌 Début de la visualisation des traces complètes...")
plot_attention_distribution("attention_weights.csv", full_dataset, trace_id=1)
print("✅ Visualisation terminée.")