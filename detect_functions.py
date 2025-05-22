import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from models.transformer1 import PowerTraceTransformer
from datasets.load_power_traces import PowerTraceDataset

def load_label_map(data_dir="Data", file_suffix="_all.csv"):
    """Charge le mapping label_id -> nom de fonction depuis le Dataset monofonction."""
    full_ds = PowerTraceDataset(data_dir, file_suffix=file_suffix)
    return full_ds.label_map


from collections import Counter

def predict_application_function(trace, model, device,
                                 window_size=512, step_size=256):
    """
    Applique le modèle à une trace d'application (1D Tensor) en la découpant en segments,
    et retourne une prédiction par segment (via argmax sur les logits).

    Returns:
        segment_preds (List[int]): liste des classes prédites pour chaque segment.
    """
    L = trace.size(0)

    model.eval()
    segment_preds = []

    with torch.no_grad():
        # Fenêtres glissantes
        for start in range(0, L - window_size + 1, step_size):
            seg = trace[start:start + window_size]
            x = seg.unsqueeze(0).unsqueeze(-1)  # (1, window_size, 1)
            mask = torch.ones(1, window_size, dtype=torch.bool, device=device)

            out = model(x, mask).squeeze(0)  # (num_classes,)
            pred_class = out.argmax().item()
            segment_preds.append(pred_class)

        # Fenêtre finale si reste
        if L % window_size != 0:
            last = trace[-window_size:]
            x = last.unsqueeze(0).unsqueeze(-1)
            mask = torch.zeros(1, window_size, dtype=torch.bool, device=device)
            valid = L % window_size
            mask[0, -valid:] = True

            out = model(x, mask).squeeze(0)
            pred_class = out.argmax().item()
            segment_preds.append(pred_class)

    if not segment_preds:
        raise ValueError(f"Trace trop courte pour taille de fenêtre {window_size}")

    return segment_preds


if __name__ == "__main__":
    # 1. Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map = load_label_map("Data")
    num_classes = len(label_map)
    model = PowerTraceTransformer(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("weights/best_model_bis.pth", map_location=device))

    # 2. Charger le CSV d'application contenant plusieurs traces
    app_file = os.path.join("Data", "Applications", "malfre_app.csv")
    try:
        df = pd.read_csv(app_file, on_bad_lines='skip')
    except pd.errors.ParserError as e:
        print(f"Erreur de parsing CSV : {e}")
        exit(1)

    df = df.select_dtypes(include=[np.number]).dropna(axis=1)
    arr = df.to_numpy(dtype=np.float32)

    # 3. Prétraitement et prédiction pour chaque trace
    for idx, trace in enumerate(arr):
        trace_norm = (trace - trace.mean()) / (trace.std() + 1e-8)
        trace_tensor = torch.tensor(trace_norm, dtype=torch.float32).to(device)
        segments = predict_application_function(trace_tensor, model, device)

        for seg in segments:
            func_name = label_map[seg]
            print(f"Trace {idx+1:>2d}: fonction prédite = '{func_name}'")