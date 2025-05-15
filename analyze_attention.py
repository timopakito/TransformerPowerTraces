import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets.load_power_traces import PowerTraceDataset

def plot_attention_distribution(csv_path, dataset, trace_id):
    df = pd.read_csv(csv_path)
    
    # Chercher la ligne correspondant à la trace
    row = df[df["trace_id"] == trace_id].iloc[0]
    label = row["label"]
    class_name = dataset.label_map[label]

    # Charger le signal brut
    trace, _ = dataset[trace_id]
    trace = trace.squeeze(-1).numpy()
    
    # Charger et parser les poids d’attention
    attention_weights = json.loads(row["attention_weights"])

    # Aplatir tous les poids pour histogramme
    all_weights = np.concatenate(attention_weights)

    # Histogramme
    plt.figure(figsize=(10, 4))
    plt.hist(all_weights, bins=50, color='royalblue', alpha=0.7)
    plt.title(f"Distribution des poids d'attention — Trace ID {trace_id} (classe {class_name})")
    plt.xlabel("Valeur du poids")
    plt.ylabel("Nombre d’occurrences")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"visualizations/hist_attention_trace_{trace_id}.png")
    plt.close()
    print(f"✅ Histogramme sauvegardé : visualizations/hist_attention_trace_{trace_id}.png")

    # Reconstruction attention moyenne sur le signal
    L = len(trace)
    window_size = len(attention_weights[0])
    step_size = window_size // 2

    combined_attention = np.zeros(L)
    overlap_counter = np.zeros(L)

    for i, window_attention in enumerate(attention_weights):
        start = i * step_size
        end = min(start + window_size, L)
        a = np.array(window_attention[:end - start])
        combined_attention[start:end] += a
        overlap_counter[start:end] += 1

    combined_attention /= np.maximum(overlap_counter, 1)

    # Courbe signal + attention
    plt.figure(figsize=(12, 4))
    plt.plot(trace, label="Signal", color='black')
    plt.plot(combined_attention, label="Attention moyenne", color='red')
    plt.title(f"Poids d'attention superposés — Trace ID {trace_id} (classe {class_name})")
    plt.xlabel("Temps")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"visualizations/curve_attention_trace_{trace_id}.png")
    plt.close()
    print(f"✅ Courbe attention sauvegardée : visualizations/curve_attention_trace_{trace_id}.png")
