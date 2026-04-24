import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

from .config import AUDIO_DIR, SAVE_DIR
from .feature_extraction import (
    fix_audio,
    extract_middle_segment,
    extract_features,
    get_feature_names,
)


def build_mir1k_unsup_dataset():
    os.makedirs(SAVE_DIR, exist_ok=True)

    wav_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(".wav")])

    X = []
    rows = []

    for wav_file in tqdm(wav_files, desc="Extracting MIR-1K features"):
        full_path = os.path.join(AUDIO_DIR, wav_file)

        try:
            y, sr = librosa.load(full_path, sr=None, mono=False)

            y = fix_audio(y, sr)
            y = extract_middle_segment(y, segment_seconds=10)

            feat = extract_features(y)

            X.append(feat)
            rows.append({
                "file_name": wav_file,
                "file_path": os.path.join("Wavfile", wav_file),
                "duration_sec_used": len(y) / 22050
            })

        except Exception as e:
            print(f"Failed: {full_path} | {e}")

    if len(X) == 0:
        raise RuntimeError("No usable WAV files found.")

    X = np.vstack(X)
    meta = pd.DataFrame(rows)
    feature_names = np.array(get_feature_names(), dtype=object)

    meta.to_csv(os.path.join(SAVE_DIR, "metadata.csv"), index=False)

    np.savez(
        os.path.join(SAVE_DIR, "features.npz"),
        X=X,
        feature_names=feature_names
    )

    print("Saved:")
    print(os.path.join(SAVE_DIR, "metadata.csv"))
    print(os.path.join(SAVE_DIR, "features.npz"))
    print("Shape:", X.shape)