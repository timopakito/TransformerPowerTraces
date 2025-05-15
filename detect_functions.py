# detect_functions.py

import torch
import torch.nn.functional as F
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from models.transformer1 import PowerTraceTransformer
from datasets.load_power_traces import ApplicationTraceDataset
from TransformerPowerTrace.train import full_dataset
from torch.utils.data import DataLoader


def detect_functions_in_trace(model, trace_tensor, label_map, device, window_size=256, step_size=128, confidence_thresh=0.7):
    """
    Donne une power trace d'application (Tensor 1D), détecte les fonctions reconnues à l'intérieur.
    Retourne un message listant les fonctions détectées avec confiance suffisante
    et affiche un graphe avec segments colorés par fonction reconnue.
    """
    model.eval()  # Mode évaluation : désactive le dropout, etc.
    trace_tensor = trace_tensor.to(device)  # Envoie la trace sur le device (GPU ou CPU)
    L = trace_tensor.size(0)  # Longueur totale de la trace

    detected_segments = []  # Stockera les segments reconnus sous forme (start, end, label)

    with torch.no_grad():  # Pas de calcul de gradient (inférence uniquement)
        for start in range(0, L - window_size + 1, step_size):
            # Découpe un segment glissant de taille `window_size`
            segment = trace_tensor[start:start + window_size]
            segment = segment.unsqueeze(0).unsqueeze(-1)  # Devient un batch de forme (1, window_size, 1)
            mask = torch.ones(1, window_size, dtype=torch.bool).to(device)  # Masque d'attention (tout activé)

            out = model(segment, mask)  # Prédiction du modèle : logits (1, num_classes)
            probs = F.softmax(out, dim=1).squeeze(0)  # Convertit en probabilités, puis enlève la dimension batch

            top_prob = probs.max().item()  # Confiance maximale
            top_class = probs.argmax().item()  # Classe prédite

            if top_prob >= confidence_thresh:
                # Si confiance suffisante, on garde ce segment
                detected_segments.append((start, start + window_size, top_class))

    if not detected_segments:
        return "📉 Lecture de la power trace...\nAucune fonction connue détectée."

    # === Analyse des fonctions détectées ===
    detected_labels = [label for (_, _, label) in detected_segments]  # On ne garde que les labels
    function_counts = Counter(detected_labels)  # Compte les occurrences de chaque fonction
    sorted_classes = [label_map[c] for c, _ in function_counts.most_common()]  # Trie par fréquence

    # === Visualisation ===
    trace_np = trace_tensor.cpu().numpy()  # Conversion en numpy pour matplotlib
    x = np.arange(len(trace_np))  # Axe des abscisses

    plt.figure(figsize=(12, 4))
    plt.plot(x, trace_np, label="Power Trace", color="lightgray")  # Trace complète en fond

    color_map = plt.get_cmap("tab10")  # Palette de couleurs (une par fonction)
    for i, (start, end, label) in enumerate(detected_segments):
        # Colorie chaque segment reconnu avec la couleur associée
        plt.plot(x[start:end], trace_np[start:end], color=color_map(label), label=label_map[label])

    # Supprimer les doublons dans la légende
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Ajouts esthétiques
    plt.title("Fonctions détectées dans la power trace")
    plt.xlabel("Temps")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("detected_functions_plot.png")  # Enregistrement
    plt.close()

    # Message utilisateur
    functions_str = " et ".join(sorted_classes)
    return f"📈 Lecture de la power trace...\nDétection des fonctions : {functions_str}. (graphe sauvegardé dans 'detected_functions_plot.png')"

#Charger le modèle entraîné
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PowerTraceTransformer()
model.load_state_dict(torch.load('weights/best_model_bis.pth', map_location=device))
model = model.to(device)

app_dataset = ApplicationTraceDataset("Data/Power_consumption_application")
app_loader = DataLoader(app_dataset, batch_size=1, shuffle=False)
label_map = full_dataset.label_map  # si tu as déjà chargé le dataset labellisé

# Exemple : appliquer le détecteur
for trace_tensor in app_loader:
    trace_tensor = trace_tensor.squeeze(0)  # (T,)
    message = detect_functions_in_trace(model, trace_tensor, label_map, device)
    print(message)


