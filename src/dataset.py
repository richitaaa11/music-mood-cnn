import os
import numpy as np
import torch
from torch.utils.data import Dataset
from config import MEL_DIR

class MusicDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.samples = []

        for _, row in self.df.iterrows():
            song_id = int(row["song_id"])
            mel_path = os.path.join(MEL_DIR, f"{song_id}.npy")
            segments = np.load(mel_path)

            for mel in segments:
                self.samples.append((mel, row["label_idx"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mel, label = self.samples[idx]

        if self.augment:
            mel = mel + np.random.normal(0, 0.01, mel.shape)

        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)

        return mel, label
