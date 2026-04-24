import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

from .config_hard import (
    AUDIO_DIR, LYRICS_DIR, SAVE_DIR,
    TARGET_SR, MIDDLE_SECONDS,
    N_MELS, N_FRAMES
)


def get_singer_id(file_name):
    return file_name.split("_")[0]


def load_audio_middle(path):
    y, sr = librosa.load(path, sr=None, mono=False)

    if y.ndim == 2:
        y = y.mean(axis=0)

    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

    mid_len = int(TARGET_SR * MIDDLE_SECONDS)

    if len(y) >= mid_len:
        start = (len(y) - mid_len) // 2
        y = y[start:start + mid_len]
    else:
        pad_total = mid_len - len(y)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        y = np.pad(y, (pad_left, pad_right))

    y = y / (np.max(np.abs(y)) + 1e-8)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    return y.astype(np.float32)


def mel_image(y):
    S = librosa.feature.melspectrogram(
        y=y,
        sr=TARGET_SR,
        n_mels=N_MELS,
        hop_length=256,
        n_fft=1024,
        power=2.0
    )

    S = librosa.power_to_db(S + 1e-10, ref=np.max)

    if S.shape[1] >= N_FRAMES:
        S = S[:, :N_FRAMES]
    else:
        pad = N_FRAMES - S.shape[1]
        S = np.pad(S, ((0, 0), (0, pad)), mode="constant")

    S = (S - S.min()) / (S.max() - S.min() + 1e-8)
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

    return S.astype(np.float32)


def read_lyrics(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def build_hard_dataset():
    os.makedirs(SAVE_DIR, exist_ok=True)

    wav_files = sorted([
        f for f in os.listdir(AUDIO_DIR)
        if f.lower().endswith(".wav")
    ])

    X_audio = []
    lyrics_list = []
    rows = []

    for wav_file in tqdm(wav_files, desc="Building hard dataset"):
        wav_path = os.path.join(AUDIO_DIR, wav_file)
        txt_file = os.path.splitext(wav_file)[0] + ".txt"
        txt_path = os.path.join(LYRICS_DIR, txt_file)

        if not os.path.exists(txt_path):
            continue

        try:
            y = load_audio_middle(wav_path)
            S = mel_image(y)
            text = read_lyrics(txt_path)

            if len(text.strip()) == 0:
                continue

            X_audio.append(S)
            lyrics_list.append(text)

            rows.append({
                "file_name": wav_file,
                "singer_id": get_singer_id(wav_file),
                "audio_path": os.path.join("Wavfile", wav_file),
                "lyrics_path": os.path.join("Lyrics", txt_file),
            })

        except Exception as e:
            print(f"Failed {wav_file}: {e}")

    if len(X_audio) == 0:
        raise RuntimeError("No valid audio-lyrics pairs found.")

    X_audio = np.array(X_audio, dtype=np.float32)
    X_audio = np.expand_dims(X_audio, axis=-1)

    metadata = pd.DataFrame(rows)

    np.save(os.path.join(SAVE_DIR, "audio_mels.npy"), X_audio)
    metadata.to_csv(os.path.join(SAVE_DIR, "metadata.csv"), index=False)

    with open(os.path.join(SAVE_DIR, "lyrics.txt"), "w", encoding="utf-8") as f:
        for line in lyrics_list:
            f.write(line.replace("\n", " ") + "\n")

    print("Saved hard dataset")
    print("Audio:", X_audio.shape)
    print("Metadata:", metadata.shape)