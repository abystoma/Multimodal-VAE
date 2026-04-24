import os
import re
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

from .config_medium import (
    AUDIO_DIR, LYRICS_DIR, SAVE_DIR,
    TARGET_SR, MIDDLE_SECONDS,
    N_MELS, N_FRAMES
)


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

    s_min, s_max = S.min(), S.max()
    S = (S - s_min) / (s_max - s_min + 1e-8)

    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

    return S.astype(np.float32)


def clean_lyrics(text):
    text = text.replace("\x00", " ")
    text = text.replace("\ufeff", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_lyrics(path):
    encodings = ["big5", "gb18030", "utf-8", "latin1"]

    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                text = f.read()
            text = clean_lyrics(text)

            # reject heavily corrupted text
            if len(text) > 0 and text.count("�") < 3:
                return text, enc

        except UnicodeDecodeError:
            continue
        except Exception:
            continue

    # final fallback
    with open(path, "r", encoding="big5", errors="ignore") as f:
        text = clean_lyrics(f.read())

    return text, "big5_ignore"


def build_mir1k_medium_dataset():
    os.makedirs(SAVE_DIR, exist_ok=True)

    wav_files = sorted([
        f for f in os.listdir(AUDIO_DIR)
        if f.lower().endswith(".wav")
    ])

    X_audio = []
    lyrics_list = []
    rows = []

    encoding_counts = {}

    for wav_file in tqdm(wav_files, desc="Building MIR-1K medium dataset"):
        wav_path = os.path.join(AUDIO_DIR, wav_file)

        txt_file = os.path.splitext(wav_file)[0] + ".txt"
        txt_path = os.path.join(LYRICS_DIR, txt_file)

        if not os.path.exists(txt_path):
            continue

        try:
            y = load_audio_middle(wav_path)
            S = mel_image(y)

            text, enc_used = read_lyrics(txt_path)

            if len(text.strip()) == 0:
                continue

            if np.isnan(S).any() or np.isinf(S).any():
                continue

            X_audio.append(S)
            lyrics_list.append(text)

            encoding_counts[enc_used] = encoding_counts.get(enc_used, 0) + 1

            rows.append({
                "file_name": wav_file,
                "duration_sec_used": len(y) / TARGET_SR,
                "audio_path": os.path.join("Wavfile", wav_file),
                "lyrics_path": os.path.join("Lyrics", txt_file),
                "lyrics_encoding": enc_used,
                "lyrics_num_chars": len(text),
            })

        except Exception as e:
            print(f"Failed {wav_file}: {e}")

    if len(X_audio) == 0:
        raise RuntimeError("No valid audio-lyrics pairs found.")

    X_audio = np.array(X_audio, dtype=np.float32)
    X_audio = np.expand_dims(X_audio, axis=-1)

    metadata = pd.DataFrame(rows)

    metadata.to_csv(os.path.join(SAVE_DIR, "metadata.csv"), index=False)
    np.save(os.path.join(SAVE_DIR, "audio_mels.npy"), X_audio)

    with open(os.path.join(SAVE_DIR, "lyrics.txt"), "w", encoding="utf-8") as f:
        for line in lyrics_list:
            f.write(line.replace("\n", " ") + "\n")

    print("\nSaved:")
    print("Metadata:", os.path.join(SAVE_DIR, "metadata.csv"))
    print("Audio:", os.path.join(SAVE_DIR, "audio_mels.npy"))
    print("Lyrics:", os.path.join(SAVE_DIR, "lyrics.txt"))
    print("Audio shape:", X_audio.shape)
    print("Audio min/max:", X_audio.min(), X_audio.max())
    print("Any NaN:", np.isnan(X_audio).any())
    print("Encoding counts:", encoding_counts)