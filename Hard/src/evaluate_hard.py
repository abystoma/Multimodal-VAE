import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    normalized_mutual_info_score,
    adjusted_rand_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

from .config_hard import SAVE_DIR, RESULT_DIR, PLOT_DIR, RANDOM_SEED


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.style.use("seaborn-v0_8-whitegrid")


TOP_N_SINGERS = 5
AUDIO_WEIGHT = 1.0
LYRICS_WEIGHT = 0.3
NOISE_SCALE = 0.05


def cluster_purity(y_true, y_pred):
    total = 0
    for c in np.unique(y_pred):
        idx = y_pred == c
        labels, counts = np.unique(y_true[idx], return_counts=True)
        total += counts.max()
    return total / len(y_true)


def run_umap(X, random_state=42):
    import umap
    return umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=random_state
    ).fit_transform(X)


def plot_embedding_with_singers(Z, labels, y_true, label_encoder, title, path):
    plt.figure(figsize=(6.2, 5.0))
    cmap = plt.get_cmap("tab10")

    for i, c in enumerate(np.unique(labels)):
        idx = labels == c

        plt.scatter(
            Z[idx, 0],
            Z[idx, 1],
            s=18,
            alpha=0.85,
            color=cmap(i % 10),
            label=f"C{c}"
        )

        singers = label_encoder.inverse_transform(y_true[idx])
        vc = pd.Series(singers).value_counts()
        top_singer = vc.index[0]
        top_count = vc.iloc[0]

        cx = Z[idx, 0].mean()
        cy = Z[idx, 1].mean()

        plt.text(
            cx,
            cy,
            f"C{c}\n{top_singer}\n({top_count})",
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.75,
                edgecolor="gray"
            )
        )

    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(frameon=True, loc="best")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def balance_with_vae_latent(z, lyrics, y):
    rng = np.random.default_rng(RANDOM_SEED)

    classes, counts = np.unique(y, return_counts=True)
    target = counts.max()

    z_out, l_out, y_out, flags = [], [], [], []

    for cls in classes:
        idx = np.where(y == cls)[0]

        z_cls = z[idx]
        l_cls = lyrics[idx]

        z_out.append(z_cls)
        l_out.append(l_cls)
        y_out.append(np.full(len(idx), cls))
        flags.append(np.zeros(len(idx), dtype=int))

        need = target - len(idx)

        if need > 0:
            sample_idx = rng.choice(idx, size=need, replace=True)

            z_std = np.std(z_cls, axis=0) + 1e-6
            l_std = np.std(l_cls, axis=0) + 1e-6

            z_synth = z[sample_idx] + rng.normal(
                0,
                NOISE_SCALE,
                size=z[sample_idx].shape
            ) * z_std

            l_synth = lyrics[sample_idx] + rng.normal(
                0,
                NOISE_SCALE,
                size=lyrics[sample_idx].shape
            ) * l_std

            z_out.append(z_synth)
            l_out.append(l_synth)
            y_out.append(np.full(need, cls))
            flags.append(np.ones(need, dtype=int))

    return (
        np.vstack(z_out).astype(np.float32),
        np.vstack(l_out).astype(np.float32),
        np.concatenate(y_out),
        np.concatenate(flags)
    )


def evaluate_representation(X, y_true, name, k):
    labels = KMeans(
        n_clusters=k,
        random_state=RANDOM_SEED,
        n_init=20
    ).fit_predict(X)

    return {
        "method": name,
        "n_clusters": k,
        "silhouette_score": silhouette_score(X, labels),
        "NMI": normalized_mutual_info_score(y_true, labels),
        "ARI": adjusted_rand_score(y_true, labels),
        "purity": cluster_purity(y_true, labels)
    }, labels


def save_cluster_singer_mapping(labels, y_true, label_encoder, path):
    rows = []

    for c in np.unique(labels):
        idx = labels == c
        singers = label_encoder.inverse_transform(y_true[idx])

        vc = pd.Series(singers).value_counts()

        rows.append({
            "cluster": c,
            "dominant_singer": vc.index[0],
            "dominant_count": int(vc.iloc[0]),
            "cluster_size": int(idx.sum()),
            "top_singers": ", ".join(vc.head(3).index.tolist()),
            "top_counts": ", ".join(map(str, vc.head(3).values.tolist()))
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)

    print("\nCluster → Singer mapping:")
    print(df)

    return df


def evaluate_hard_balanced():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    z = np.load(os.path.join(RESULT_DIR, "z_mean_beta.npy"))
    lyrics = np.load(os.path.join(SAVE_DIR, "lyrics_embed.npy"))
    meta = pd.read_csv(os.path.join(SAVE_DIR, "metadata.csv"))

    top_singers = meta["singer_id"].value_counts().head(TOP_N_SINGERS).index
    mask = meta["singer_id"].isin(top_singers).values

    meta_f = meta.loc[mask].reset_index(drop=True)
    z = z[mask]
    lyrics = lyrics[mask]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(meta_f["singer_id"])

    print("Top singers used:")
    print(meta_f["singer_id"].value_counts())

    z, lyrics, y, flags = balance_with_vae_latent(z, lyrics, y)

    print("\nBalanced class counts:")
    print(pd.Series(y).value_counts().sort_index())
    print("Original samples:", int(np.sum(flags == 0)))
    print("Synthetic samples:", int(np.sum(flags == 1)))

    X = np.concatenate(
        [
            AUDIO_WEIGHT * StandardScaler().fit_transform(z),
            LYRICS_WEIGHT * StandardScaler().fit_transform(lyrics)
        ],
        axis=1
    )

    X = StandardScaler().fit_transform(X)

    X_beta = PCA(
        n_components=8,
        random_state=RANDOM_SEED
    ).fit_transform(X)

    X_pca = PCA(
        n_components=16,
        random_state=RANDOM_SEED
    ).fit_transform(X)

    k = len(np.unique(y))

    res1, labels_beta = evaluate_representation(
        X_beta,
        y,
        "Balanced Beta-VAE Hybrid + PCA(8) + KMeans",
        k
    )

    res2, labels_pca = evaluate_representation(
        X_pca,
        y,
        "Balanced PCA Hybrid + KMeans",
        k
    )

    results = pd.DataFrame([res1, res2])
    results.to_csv(
        os.path.join(RESULT_DIR, "hard_results_balanced.csv"),
        index=False
    )

    assignments = pd.DataFrame({
        "true_label": y,
        "true_singer": label_encoder.inverse_transform(y),
        "synthetic": flags,
        "beta_cluster": labels_beta,
        "pca_cluster": labels_pca
    })

    assignments.to_csv(
        os.path.join(RESULT_DIR, "cluster_assignments.csv"),
        index=False
    )

    save_cluster_singer_mapping(
        labels_beta,
        y,
        label_encoder,
        os.path.join(RESULT_DIR, "cluster_singer_mapping_beta.csv")
    )

    save_cluster_singer_mapping(
        labels_pca,
        y,
        label_encoder,
        os.path.join(RESULT_DIR, "cluster_singer_mapping_pca.csv")
    )

    Z_beta = run_umap(X_beta, RANDOM_SEED)
    plot_embedding_with_singers(
        Z_beta,
        labels_beta,
        y,
        label_encoder,
        "Balanced Beta-VAE Hybrid PCA(8) Clusters",
        os.path.join(PLOT_DIR, "balanced_beta_vae_pca8_clusters_with_singers.pdf")
    )

    Z_pca = run_umap(X_pca, RANDOM_SEED)
    plot_embedding_with_singers(
        Z_pca,
        labels_pca,
        y,
        label_encoder,
        "Balanced PCA Hybrid Clusters",
        os.path.join(PLOT_DIR, "balanced_pca_clusters_with_singers.pdf")
    )

    print("\n===== HARD BALANCED RESULTS =====")
    print(results)

    print("\nSaved:")
    print(os.path.join(RESULT_DIR, "hard_results_balanced.csv"))
    print(os.path.join(RESULT_DIR, "cluster_assignments.csv"))
    print(os.path.join(RESULT_DIR, "cluster_singer_mapping_beta.csv"))
    print(os.path.join(RESULT_DIR, "cluster_singer_mapping_pca.csv"))