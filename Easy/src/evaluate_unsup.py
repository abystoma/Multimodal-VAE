import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from .config import SAVE_DIR, RESULT_DIR, PLOT_DIR, LATENT_DIM, RANDOM_SEED, N_CLUSTERS


plt.style.use("seaborn-v0_8-whitegrid")


def run_umap(X, random_state=42):
    import umap
    return umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=random_state
    ).fit_transform(X)


def plot_embedding(Z, labels, title, save_path):
    cmap = plt.get_cmap("tab10")

    plt.figure(figsize=(5.5, 4.5))

    for i, lab in enumerate(np.unique(labels)):
        idx = labels == lab
        plt.scatter(
            Z[idx, 0],
            Z[idx, 1],
            s=18,
            alpha=0.85,
            color=cmap(i),
            label=f"Cluster {lab}"
        )

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_unsup():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Load data
    X_scaled = np.load(os.path.join(SAVE_DIR, "X_scaled.npy"))
    z_mean = np.load(os.path.join(RESULT_DIR, "z_mean_mir1k_vae.npy"))

    print("Latent shape:", z_mean.shape)

    # ========================
    # VAE + KMeans
    # ========================
    kmeans_vae = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_SEED,
        n_init=20
    )
    pred_vae = kmeans_vae.fit_predict(z_mean)

    sil_vae = silhouette_score(z_mean, pred_vae)
    ch_vae = calinski_harabasz_score(z_mean, pred_vae)

    # ========================
    # PCA + KMeans
    # ========================
    pca = PCA(n_components=LATENT_DIM, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)

    kmeans_pca = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_SEED,
        n_init=20
    )
    pred_pca = kmeans_pca.fit_predict(X_pca)

    sil_pca = silhouette_score(X_pca, pred_pca)
    ch_pca = calinski_harabasz_score(X_pca, pred_pca)

    # ========================
    # Save results
    # ========================
    results = pd.DataFrame([
        {
            "method": "VAE + KMeans",
            "silhouette_score": sil_vae,
            "calinski_harabasz_index": ch_vae
        },
        {
            "method": "PCA + KMeans",
            "silhouette_score": sil_pca,
            "calinski_harabasz_index": ch_pca
        }
    ])

    results.to_csv(
        os.path.join(RESULT_DIR, "clustering_results_mir1k_unsup.csv"),
        index=False
    )

    print("\n===== FINAL RESULTS =====")
    print(results)

    # ========================
    # UMAP plots
    # ========================
    Z_vae = run_umap(z_mean, RANDOM_SEED)
    Z_pca = run_umap(X_pca, RANDOM_SEED)

    plot_embedding(
        Z_vae,
        pred_vae,
        "VAE Latent Clusters",
        os.path.join(PLOT_DIR, "vae_clusters.pdf")
    )

    plot_embedding(
        Z_pca,
        pred_pca,
        "PCA Clusters",
        os.path.join(PLOT_DIR, "pca_clusters.pdf")
    )