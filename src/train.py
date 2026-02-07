import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import *
from dataset import MusicDataset
from model import AudioCNN

def collate_fn(batch):
    mels, labels = zip(*batch)

    max_len = max(mel.shape[2] for mel in mels)
    mels_padded = [
        torch.nn.functional.pad(mel, (0, max_len - mel.shape[2]))
        for mel in mels
    ]

    return torch.stack(mels_padded), torch.stack(labels)


def main():
    df = pd.read_csv(f"{DEAM_PATH}/Annotations/static_annotations_averaged_songs_1_2000.csv")
    df.columns = df.columns.str.strip()

    df["label_idx"] = (df["valence_mean"] > 5).astype(int) * 2 + (df["arousal_mean"] > 5).astype(int)

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["label_idx"], random_state=42
    )

    train_ds = MusicDataset(train_df, augment=True)
    val_ds = MusicDataset(val_df)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, BATCH_SIZE, collate_fn=collate_fn)

    model = AudioCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)

        print(f"Train Acc: {correct/total:.3f}")


if __name__ == "__main__":
    main()
