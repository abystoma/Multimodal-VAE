import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from .config_hard import (
    SAVE_DIR, MODEL_DIR, PLOT_DIR, RESULT_DIR,
    LATENT_DIM, BATCH_SIZE, EPOCHS, LEARNING_RATE, RANDOM_SEED, BETA
)
from .beta_vae_hard import build_encoder, build_decoder, MultiModalBetaVAE


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.style.use("seaborn-v0_8-whitegrid")


def plot_loss(history, key, title, save_path):
    plt.figure(figsize=(5.5, 4.5))

    if key in history.history:
        plt.plot(history.history[key], label=f"train {key}")

    val_key = f"val_{key}"
    if val_key in history.history:
        plt.plot(history.history[val_key], label=f"val {key}")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def train_beta_vae():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    X_train = np.load(os.path.join(SAVE_DIR, "audio_train.npy")).astype(np.float32)
    X_val = np.load(os.path.join(SAVE_DIR, "audio_val.npy")).astype(np.float32)
    X_all = np.load(os.path.join(SAVE_DIR, "audio_all.npy")).astype(np.float32)

    L_train = np.load(os.path.join(SAVE_DIR, "lyrics_train.npy")).astype(np.float32)
    L_val = np.load(os.path.join(SAVE_DIR, "lyrics_val.npy")).astype(np.float32)
    L_all = np.load(os.path.join(SAVE_DIR, "lyrics_embed.npy")).astype(np.float32)

    print("Audio train:", X_train.shape)
    print("Lyrics train:", L_train.shape)

    encoder, shape_before_flatten = build_encoder(
        audio_shape=X_train.shape[1:],
        lyrics_dim=L_train.shape[1],
        latent_dim=LATENT_DIM
    )

    decoder = build_decoder(
        shape_before_flatten=shape_before_flatten,
        latent_dim=LATENT_DIM,
        lyrics_dim=L_train.shape[1]
    )

    vae = MultiModalBetaVAE(
        encoder,
        decoder,
        beta=BETA,
        lyrics_weight=0.5
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=1.0
    )

    vae.compile(optimizer=optimizer)

    history = vae.fit(
        (X_train, L_train),
        validation_data=(X_val, L_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    encoder.save_weights(os.path.join(MODEL_DIR, "multimodal_beta_encoder.weights.h5"))
    decoder.save_weights(os.path.join(MODEL_DIR, "multimodal_beta_decoder.weights.h5"))

    pd.DataFrame(history.history).to_csv(
        os.path.join(RESULT_DIR, "multimodal_beta_vae_history.csv"),
        index=False
    )

    z_mean, z_log_var, z = encoder.predict(
        [X_all, L_all],
        batch_size=BATCH_SIZE,
        verbose=0
    )

    np.save(os.path.join(RESULT_DIR, "z_mean_beta.npy"), z_mean)
    np.save(os.path.join(RESULT_DIR, "z_log_var_beta.npy"), z_log_var)
    np.save(os.path.join(RESULT_DIR, "z_sample_beta.npy"), z)

    plot_loss(
        history,
        "loss",
        "Multimodal Beta-VAE Total Loss",
        os.path.join(PLOT_DIR, "multimodal_beta_vae_total_loss.pdf")
    )

    plot_loss(
        history,
        "audio_recon_loss",
        "Audio Reconstruction Loss",
        os.path.join(PLOT_DIR, "multimodal_beta_audio_loss.pdf")
    )

    plot_loss(
        history,
        "lyrics_recon_loss",
        "Lyrics Reconstruction Loss",
        os.path.join(PLOT_DIR, "multimodal_beta_lyrics_loss.pdf")
    )

    plot_loss(
        history,
        "kl_loss",
        "KL Divergence",
        os.path.join(PLOT_DIR, "multimodal_beta_kl_loss.pdf")
    )

    print("Saved multimodal Beta-VAE outputs.")