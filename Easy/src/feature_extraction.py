import numpy as np
import librosa
from .config import TARGET_SR


def fix_audio(y, sr):
    if y.ndim == 2:
        y = y.mean(axis=0)

    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

    y = y / (np.max(np.abs(y)) + 1e-8)
    return y.astype(np.float32)


def extract_middle_segment(y, segment_seconds=10):
    target_len = int(TARGET_SR * segment_seconds)

    # take middle from real audio first
    if len(y) >= target_len:
        start = (len(y) - target_len) // 2
        y = y[start:start + target_len]
    else:
        # pad only after using full real audio
        pad_total = target_len - len(y)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        y = np.pad(y, (pad_left, pad_right))

    return y.astype(np.float32)


def extract_features(y):
    feats = []

    mfcc = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=20)
    feats.extend(np.mean(mfcc, axis=1))
    feats.extend(np.std(mfcc, axis=1))

    centroid = librosa.feature.spectral_centroid(y=y, sr=TARGET_SR)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=TARGET_SR)
    contrast = librosa.feature.spectral_contrast(y=y, sr=TARGET_SR)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=TARGET_SR)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    for f in [centroid, bandwidth, contrast, rolloff, zcr, rms]:
        feats.extend(np.mean(f, axis=1))
        feats.extend(np.std(f, axis=1))

    return np.array(feats, dtype=np.float32)


def get_feature_names():
    feature_names = []

    for i in range(20):
        feature_names.append(f"mfcc_{i+1}_mean")
    for i in range(20):
        feature_names.append(f"mfcc_{i+1}_std")

    feature_names += [
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_bandwidth_mean",
        "spectral_bandwidth_std",
    ]

    for i in range(7):
        feature_names.append(f"spectral_contrast_{i+1}_mean")
    for i in range(7):
        feature_names.append(f"spectral_contrast_{i+1}_std")

    feature_names += [
        "spectral_rolloff_mean",
        "spectral_rolloff_std",
        "zcr_mean",
        "zcr_std",
        "rms_mean",
        "rms_std",
    ]

    return feature_names