import numpy as np
import random
import librosa
from config import SR, N_MELS

def augment_audio(y, sr=SR):
    if random.random() < 0.5:
        rate = random.uniform(0.9, 1.1)
        y = librosa.effects.time_stretch(y=y, rate=rate)

    if random.random() < 0.5:
        n_steps = random.randint(-2, 2)
        y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

    return y


def mel_spectrogram(y):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=N_MELS,
        fmax=8000
    )
    mel = librosa.power_to_db(mel)
    return mel.astype(np.float32)
