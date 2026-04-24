import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config_hard import (
    SAVE_DIR, TEST_SIZE, RANDOM_SEED,
    LYRICS_MAX_FEATURES, LYRICS_EMBED_DIM
)


def prepare_hard_data():
    X_audio = np.load(os.path.join(SAVE_DIR, "audio_mels.npy")).astype(np.float32)
    metadata = pd.read_csv(os.path.join(SAVE_DIR, "metadata.csv"))

    with open(os.path.join(SAVE_DIR, "lyrics.txt"), "r", encoding="utf-8") as f:
        lyrics = [line.strip() for line in f.readlines()]

    if len(X_audio) != len(lyrics):
        raise ValueError("Mismatch between audio and lyrics count.")

    vectorizer = TfidfVectorizer(
        max_features=LYRICS_MAX_FEATURES,
        lowercase=True,
        stop_words="english"
    )

    X_tfidf = vectorizer.fit_transform(lyrics)

    svd = TruncatedSVD(
        n_components=LYRICS_EMBED_DIM,
        random_state=RANDOM_SEED
    )

    X_lyrics = svd.fit_transform(X_tfidf).astype(np.float32)

    scaler = StandardScaler()
    X_lyrics = scaler.fit_transform(X_lyrics).astype(np.float32)

    label_encoder = LabelEncoder()
    y_singer = label_encoder.fit_transform(metadata["singer_id"])

    idx = np.arange(len(X_audio))

    train_idx, val_idx = train_test_split(
        idx,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )

    np.save(os.path.join(SAVE_DIR, "audio_train.npy"), X_audio[train_idx])
    np.save(os.path.join(SAVE_DIR, "audio_val.npy"), X_audio[val_idx])
    np.save(os.path.join(SAVE_DIR, "audio_all.npy"), X_audio)

    np.save(os.path.join(SAVE_DIR, "lyrics_train.npy"), X_lyrics[train_idx])
    np.save(os.path.join(SAVE_DIR, "lyrics_val.npy"), X_lyrics[val_idx])
    np.save(os.path.join(SAVE_DIR, "lyrics_embed.npy"), X_lyrics)

    np.save(os.path.join(SAVE_DIR, "singer_labels.npy"), y_singer)
    np.save(os.path.join(SAVE_DIR, "train_idx.npy"), train_idx)
    np.save(os.path.join(SAVE_DIR, "val_idx.npy"), val_idx)

    pd.DataFrame({
        "singer_id": label_encoder.classes_
    }).to_csv(os.path.join(SAVE_DIR, "singer_label_map.csv"), index=False)

    print("Saved hard prepared data")
    print("Audio:", X_audio.shape)
    print("Lyrics:", X_lyrics.shape)
    print("Singer labels:", y_singer.shape)