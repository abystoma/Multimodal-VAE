import os
import re
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from .config_medium import (
    SAVE_DIR,
    TEST_SIZE,
    RANDOM_SEED,
    LYRICS_MAX_FEATURES,
    LYRICS_EMBED_DIM
)


def clean_text_for_tfidf(text):
    text = str(text)
    text = text.replace("\x00", " ")
    text = text.replace("\ufeff", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_medium_data():
    X_audio = np.load(os.path.join(SAVE_DIR, "audio_mels.npy")).astype(np.float32)

    with open(os.path.join(SAVE_DIR, "lyrics.txt"), "r", encoding="utf-8", errors="ignore") as f:
        lyrics = [clean_text_for_tfidf(line) for line in f.readlines()]

    if len(X_audio) != len(lyrics):
        raise ValueError(
            f"Mismatch: audio samples={len(X_audio)}, lyrics lines={len(lyrics)}"
        )

    # Character n-grams work better for Chinese/Mandarin lyrics than English word tokens.
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=LYRICS_MAX_FEATURES,
        lowercase=False
    )

    X_tfidf = vectorizer.fit_transform(lyrics)

    svd_dim = min(LYRICS_EMBED_DIM, X_tfidf.shape[1] - 1)

    if svd_dim < 2:
        raise RuntimeError(
            f"Too few lyric features for SVD. TF-IDF shape: {X_tfidf.shape}"
        )

    svd = TruncatedSVD(
        n_components=svd_dim,
        random_state=RANDOM_SEED
    )

    X_lyrics = svd.fit_transform(X_tfidf).astype(np.float32)

    scaler = StandardScaler()
    X_lyrics = scaler.fit_transform(X_lyrics).astype(np.float32)

    idx = np.arange(len(X_audio))

    train_idx, val_idx = train_test_split(
        idx,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )

    np.save(os.path.join(SAVE_DIR, "audio_train.npy"), X_audio[train_idx])
    np.save(os.path.join(SAVE_DIR, "audio_val.npy"), X_audio[val_idx])
    np.save(os.path.join(SAVE_DIR, "audio_all.npy"), X_audio)

    np.save(os.path.join(SAVE_DIR, "lyrics_embed.npy"), X_lyrics)
    np.save(os.path.join(SAVE_DIR, "lyrics_train.npy"), X_lyrics[train_idx])
    np.save(os.path.join(SAVE_DIR, "lyrics_val.npy"), X_lyrics[val_idx])

    np.save(os.path.join(SAVE_DIR, "train_idx.npy"), train_idx)
    np.save(os.path.join(SAVE_DIR, "val_idx.npy"), val_idx)

    print("\nSaved:")
    print(os.path.join(SAVE_DIR, "audio_train.npy"))
    print(os.path.join(SAVE_DIR, "audio_val.npy"))
    print(os.path.join(SAVE_DIR, "audio_all.npy"))
    print(os.path.join(SAVE_DIR, "lyrics_embed.npy"))

    print("\nShapes:")
    print("Audio all:", X_audio.shape)
    print("TF-IDF:", X_tfidf.shape)
    print("Lyrics embed:", X_lyrics.shape)

    print("\nLyrics embedding stats:")
    print("Mean:", X_lyrics.mean())
    print("Std:", X_lyrics.std())
    print("Any NaN:", np.isnan(X_lyrics).any())