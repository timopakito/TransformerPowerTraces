import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
from collections import defaultdict

class PowerTraceDataset(Dataset):
    def __init__(self, data_dir, file_suffix="_all.csv", normalize=True):
        self.traces = []
        self.labels = []
        self.label_map = {}
        
        # Lister les fichiers
        files = sorted([f for f in os.listdir(data_dir) if f.endswith(file_suffix)])

        for label_id, filename in enumerate(files):
            function_name = filename.replace(file_suffix, "")
            self.label_map[label_id] = function_name

            print(f"Chargement : {filename} → label {label_id} ({function_name})")

            # Charger le fichier
            df = pd.read_csv(os.path.join(data_dir, filename))
            df = df.select_dtypes(include=[np.number])
            df = df.dropna(axis=1, how='any')
            arr = df.to_numpy(dtype=np.float32)  # shape: (n_traces, T)
            lengths = [len(trace) for trace in arr]
            print(f"Nombre de traces = {len(lengths)}")
            print(f"Longueur min = {min(lengths)}, max = {max(lengths)}, moyenne = {sum(lengths)//len(lengths)}")

            if normalize:
                arr = (arr - arr.mean(axis=1, keepdims=True)) / (arr.std(axis=1, keepdims=True) + 1e-8)

            # Ajouter chaque trace individuellement
            for trace in arr:
                self.traces.append(torch.tensor(trace, dtype=torch.float32))
                self.labels.append(label_id)

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        # Retourne (T, 1), label
        return self.traces[idx].unsqueeze(-1), self.labels[idx]


# Fonction pour coller les batches et gérer le padding
def collate_fn(batch):
    sequences, labels = zip(*batch)  # chaque sequence a taille (Ti, 1)
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0.0)  # (B, T_max, 1)
    lengths = torch.tensor([seq.size(0) for seq in sequences])
    attention_mask = torch.arange(padded_seqs.size(1)).unsqueeze(0) < lengths.unsqueeze(1)  # (B, T_max)
    return padded_seqs, torch.tensor(labels), attention_mask

class PowerTraceDatasetSlidingWindow(Dataset):
    def __init__(self, data_dir, file_suffix="_all.csv", normalize=True, window_size=512, step_size=256):
        self.traces = []
        self.masks = []
        self.labels = []
        self.label_map = {}
        self.window_size = window_size
        self.step_size = step_size

        # Lister les fichiers dans le dossier
        files = sorted([f for f in os.listdir(data_dir) if f.endswith(file_suffix)])

        for label_id, filename in enumerate(files):
            function_name = filename.replace(file_suffix, "")
            self.label_map[label_id] = function_name

            print(f"Chargement : {filename} → label {label_id} ({function_name})")

            # Chargement des données
            df = pd.read_csv(os.path.join(data_dir, filename))
            df = df.select_dtypes(include=[np.number])
            df = df.dropna(axis=1, how='any')
            arr = df.to_numpy(dtype=np.float32)

            # Normalisation
            if normalize:
                arr = (arr - arr.mean(axis=1, keepdims=True)) / (arr.std(axis=1, keepdims=True) + 1e-8)

            # Fenêtrage glissant
            for trace in arr:
                windows, masks = self.sliding_window(trace)
                self.traces.extend(windows)
                self.masks.extend(masks)
                self.labels.extend([label_id] * len(windows))
    
    def sliding_window(self, trace):
        """ Découpe une trace en segments de taille fixe avec un overlap. """
        L = len(trace)
        windows, masks = [], []
        
        # Fenêtrage
        for start in range(0, L - self.window_size + 1, self.step_size):
            segment = trace[start:start + self.window_size]
            windows.append(torch.tensor(segment, dtype=torch.float32).unsqueeze(-1))
            masks.append(torch.ones(self.window_size, dtype=torch.bool))
        
        # Si la dernière portion est trop courte
        if (L % self.window_size) != 0:
            last_segment = trace[-self.window_size:]
            padded_segment = torch.zeros(self.window_size)
            padded_segment[-len(last_segment):] = torch.tensor(last_segment, dtype=torch.float32)
            windows.append(padded_segment.unsqueeze(-1))
            
            mask = torch.zeros(self.window_size, dtype=torch.bool)
            mask[-len(last_segment):] = True
            masks.append(mask)
        
        return windows, masks

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        return self.traces[idx], self.labels[idx], self.masks[idx]

def collate_fn_sliding_window(batch):
    traces, labels, masks = zip(*batch)

    traces = torch.stack(traces)  # (B, 512, 1)
    labels = torch.tensor(labels, dtype=torch.long)

    # Sécurité : si mask est un booléen numpy ou liste, on force en tensor
    masks = torch.stack([
        torch.tensor(m, dtype=torch.bool) if not isinstance(m, torch.Tensor) else m.to(torch.bool)
        for m in masks
    ])

    return traces, labels, masks

class SlidingWindowWrapper(Dataset):
    def __init__(self, base_dataset, window_size=512, step_size=256):
        self.base_dataset = base_dataset
        self.window_size = window_size
        self.step_size = step_size
        self.windowed_traces = []
        self.labels = []

        for idx in range(len(base_dataset)):
            trace, label = base_dataset[idx]
            trace = trace.squeeze(-1)  # (T,)
            windows = self.sliding_window(trace)
            self.windowed_traces.extend(windows)
            self.labels.extend([label] * len(windows))

    def sliding_window(self, trace):
        L = trace.shape[0]
        windows = []
        for start in range(0, L - self.window_size + 1, self.step_size):
            segment = trace[start:start + self.window_size]
            windows.append(segment.unsqueeze(-1))  # (T, 1)
        return windows

    def __len__(self):
        return len(self.windowed_traces)

    def __getitem__(self, idx):
        trace = self.windowed_traces[idx]
        label = self.labels[idx]
        mask = torch.ones(self.window_size, dtype=torch.bool)
        return trace, label, mask 
    
def collate_fn_val(batch):
    sequences, labels = zip(*batch)  # batch_size = 1
    return sequences[0], labels[0]

class PaddedShortTraceDataset(Dataset):
    def __init__(self, dataset, max_len=512):
        self.samples = []
        for x, y in dataset:
            # Ne conserve que les traces courtes
            if x.size(0) < max_len:
                padded = torch.zeros(max_len)
                padded[:x.size(0)] = x.squeeze(-1)

                mask = torch.zeros(max_len, dtype=torch.bool)
                mask[:x.size(0)] = True

                self.samples.append((
                    padded.unsqueeze(-1),  # (512, 1)
                    int(y),                # Cast explicite en int natif ✅
                    mask                   # (512,) attention mask
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
def collate_fn_unified(batch):
    """
    Batch a list of (trace, label, mask) tuples.
    - trace: Tensor (T, 1)
    - label: int or scalar Tensor
    - mask: Tensor (T,)
    """
    traces, labels, masks = zip(*batch)

    # Empile les traces (T, 1) → (B, T, 1)
    traces = torch.stack(traces)

    # Assure que tous les labels sont des int natifs avant conversion
    labels = torch.tensor([
    l.item() if isinstance(l, torch.Tensor) else int(l)
    for l in labels
], dtype=torch.long)

    # Assure que tous les masques sont des BoolTensors
    masks = torch.stack([
        torch.tensor(m, dtype=torch.bool) if not isinstance(m, torch.Tensor) else m.to(torch.bool)
        for m in masks
    ])

    return traces, labels, masks

class ApplicationTraceDataset(Dataset):
    def __init__(self, data_dir, file_suffix=".csv", normalize=True):
        """
        Dataset pour les power traces d'application non labellisées.

        Args:
            data_dir (str): Chemin vers le dossier contenant les fichiers CSV.
            file_suffix (str): Suffixe des fichiers CSV (ex: "_app.csv").
            normalize (bool): Appliquer une normalisation par trace.
        """
        self.traces = []

        files = sorted([f for f in os.listdir(data_dir) if f.endswith(file_suffix)])
        print(f"📦 Fichiers chargés : {files}")

        for filename in files:
            try:
                df = pd.read_csv(os.path.join(data_dir, filename), on_bad_lines='skip')
            except Exception as e:
                print(f"❌ Erreur dans {filename} : {e}")
                continue

            df = df.select_dtypes(include=["number"]).dropna(axis=1)
            arr = df.to_numpy(dtype=np.float32)

            for trace in arr:
                if normalize:
                    trace = (trace - trace.mean()) / (trace.std() + 1e-8)

                tensor = torch.tensor(trace, dtype=torch.float32)
                self.traces.append(tensor)

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        return self.traces[idx]  # (T,)

class BalancedSegmentDataset(Dataset):
    def __init__(self, datasets, segments_per_class):
        """
        datasets : liste de datasets contenant chacun des tuples (trace, label, mask)
        segments_per_class : int, nombre de segments à conserver pour chaque classe
        """
        self.samples = []
        class_buckets = defaultdict(list)

        # Regroupe tous les segments par classe
        for dataset in datasets:
            for trace, label, mask in dataset:
                class_buckets[label].append((trace, label, mask))

        # Pour chaque classe, échantillonne les segments
        for label, samples in class_buckets.items():
            if len(samples) >= segments_per_class:
                selected = random.sample(samples, segments_per_class)  # Sous-échantillonnage
            else:
                # Sur-échantillonnage si classe minoritaire
                selected = random.choices(samples, k=segments_per_class)
            self.samples.extend(selected)

        random.shuffle(self.samples)  # Mélange pour éviter les blocs par classe

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
