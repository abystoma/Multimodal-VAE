import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from .config import (
    SAVE_DIR, MODEL_DIR, PLOT_DIR, RESULT_DIR,
    LATENT_DIM, BATCH_SIZE, EPOCHS, LEARNING_RATE, RANDOM_SEED
)
from .vae_model import build_encoder, build_decoder, VAE


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9
})


def plot_loss(history, key, title, save_path):
    plt.figure(figsize=(5.5, 4.5))
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


def train_vae_unsup():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    X_train = np.load(os.path.join(SAVE_DIR, "X_train.npy")).astype(np.float32)
    X_val = np.load(os.path.join(SAVE_DIR, "X_val.npy")).astype(np.float32)
    X_scaled = np.load(os.path.join(SAVE_DIR, "X_scaled.npy")).astype(np.float32)

    input_dim = X_train.shape[1]

    encoder = build_encoder(input_dim=input_dim, latent_dim=LATENT_DIM)
    decoder = build_decoder(input_dim=input_dim, latent_dim=LATENT_DIM)
    vae = VAE(encoder, decoder, beta=0.01)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))

    history = vae.fit(
        X_train,
        validation_data=(X_val,),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    encoder.save_weights(os.path.join(MODEL_DIR, "encoder_mir1k_vae.weights.h5"))
    decoder.save_weights(os.path.join(MODEL_DIR, "decoder_mir1k_vae.weights.h5"))

    pd.DataFrame(history.history).to_csv(
        os.path.join(RESULT_DIR, "vae_history_mir1k_unsup.csv"), index=False
    )

    z_mean, z_log_var, z = encoder.predict(X_scaled, batch_size=BATCH_SIZE, verbose=0)
    np.save(os.path.join(RESULT_DIR, "z_mean_mir1k_vae.npy"), z_mean)
    np.save(os.path.join(RESULT_DIR, "z_log_var_mir1k_vae.npy"), z_log_var)
    np.save(os.path.join(RESULT_DIR, "z_sample_mir1k_vae.npy"), z)

    print("Saved latent:", os.path.join(RESULT_DIR, "z_mean_mir1k_vae.npy"))
    print("z_mean shape:", z_mean.shape)
    print("z_mean sample:", z_mean[:2])

    plot_loss(
        history, "loss", "VAE Total Loss",
        os.path.join(PLOT_DIR, "vae_total_loss_mir1k_unsup.pdf")
    )
    plot_loss(
        history, "recon_loss", "VAE Reconstruction Loss",
        os.path.join(PLOT_DIR, "vae_recon_loss_mir1k_unsup.pdf")
    )
    plot_loss(
        history, "kl_loss", "VAE KL Divergence",
        os.path.join(PLOT_DIR, "vae_kl_loss_mir1k_unsup.pdf")
    )

    print("Saved Normal VAE outputs.")