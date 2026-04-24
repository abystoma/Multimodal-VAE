import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import SAVE_DIR, TEST_SIZE, RANDOM_SEED


def prepare_data_unsup():
    data = np.load(os.path.join(SAVE_DIR, "features.npz"), allow_pickle=True)
    X = data["X"].astype(np.float32)
    feature_names = data["feature_names"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    X_train, X_val = train_test_split(
        X_scaled,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )

    np.save(os.path.join(SAVE_DIR, "X_scaled.npy"), X_scaled)
    np.save(os.path.join(SAVE_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(SAVE_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(SAVE_DIR, "feature_names.npy"), feature_names, allow_pickle=True)

    print("Saved:")
    print(os.path.join(SAVE_DIR, "X_scaled.npy"))
    print(os.path.join(SAVE_DIR, "X_train.npy"))
    print(os.path.join(SAVE_DIR, "X_val.npy"))
    print(os.path.join(SAVE_DIR, "feature_names.npy"))
    print("X_scaled:", X_scaled.shape)
    print("Mean:", X_scaled.mean(), "Std:", X_scaled.std())