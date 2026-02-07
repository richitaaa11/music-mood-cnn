import os
import torch

# Paths 
DEAM_PATH = "C:\Users\richita\Documents\DEAM"
AUDIO_DIR = os.path.join(DEAM_PATH, "Audio")
MEL_DIR   = os.path.join(DEAM_PATH, "MEL")

# Audio parameters
SR = 22050
N_MELS = 128
WINDOW_SEC = 20
HOP_SEC = 10

# Training parameters
BATCH_SIZE = 4
EPOCHS = 5
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
