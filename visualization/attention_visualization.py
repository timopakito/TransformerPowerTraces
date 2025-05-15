import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd
import numpy as np
import os
import json
from datasets.load_power_traces import PowerTraceDataset

def visualize_attention_signal(dataset, csv_path="attention_weights.csv", save_dir="visualizations", n_samples=3):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df = pd.read_csv(csv_path)
    label_map = dataset.label_map

    for class_id, class_name in label_map.items():
        samples = df[df['label'] == class_id].head(n_samples)
        
        for sample_idx, sample in samples.iterrows():
            trace, _ = dataset[int(sample['trace_id'])]
            trace = trace.squeeze(-1).numpy()

            try:
                attention_weights = json.loads(sample['attention_weights'])
            except Exception as e:
                print(f"⚠️ Erreur de parsing JSON pour trace {sample['trace_id']}: {e}")
                continue

            if not attention_weights or not isinstance(attention_weights[0], list):
                print(f"⚠️ Attention invalide pour trace {sample['trace_id']}")
                continue

            # === Reconstruction attention moyenne
            L = len(trace)
            window_size = len(attention_weights[0])
            step_size = window_size // 2

            combined_attention = np.zeros(L)
            overlap_counter = np.zeros(L)

            for i, window_attention in enumerate(attention_weights):
                start = i * step_size
                end = min(start + window_size, L)
                weights = np.array(window_attention[:end - start])
                combined_attention[start:end] += weights
                overlap_counter[start:end] += 1

            combined_attention /= np.maximum(overlap_counter, 1)

            # === Création de segments colorés
            x = np.arange(L)
            y = trace
            attention = combined_attention

            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            norm = plt.Normalize(attention.min(), attention.max())
            lc = LineCollection(segments, cmap='hot', norm=norm)
            lc.set_array(attention[:-1])
            lc.set_linewidth(2)

            # === Affichage
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.add_collection(lc)
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(y.min(), y.max())
            ax.set_title(f"Trace ID {sample['trace_id']} – Classe {class_name}")
            ax.set_xlabel("Temps")
            ax.set_ylabel("Amplitude signal")
            plt.colorbar(lc, ax=ax, label="Intensité d'attention")

            # === Sauvegarde
            filename = f"{save_dir}/{class_name}_trace_{sample['trace_id']}_colored.png"
            plt.savefig(filename)
            plt.close()
            print(f"✅ Sauvegardé : {filename}")


full_dataset = PowerTraceDataset("Data")
visualize_attention_signal(full_dataset, csv_path="attention_weights.csv", save_dir="visualizations")
