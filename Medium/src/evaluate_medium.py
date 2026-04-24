import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .config_medium import SAVE_DIR, RESULT_DIR, PLOT_DIR, RANDOM_SEED, N_CLUSTERS


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.style.use("seaborn-v0_8-whitegrid")


DBSCAN_PCA_DIM = 10
DBSCAN_MIN_SAMPLES = 5
DBSCAN_EPS_VALUES = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]


def run_umap(X, random_state=42):
    import umap

    return umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=random_state
    ).fit_transform(X)


def safe_metrics(X, labels):
    mask = labels != -1

    if mask.sum() < 2:
        return np.nan, np.nan

    X_eval = X[mask]
    y_eval = labels[mask]

    if len(np.unique(y_eval)) < 2:
        return np.nan, np.nan

    sil = silhouette_score(X_eval, y_eval)
    dbi = davies_bouldin_score(X_eval, y_eval)

    return sil, dbi


def plot_embedding(Z, labels, title, save_path):
    cmap = plt.get_cmap("tab10")

    plt.figure(figsize=(5.5, 4.5))

    for i, lab in enumerate(np.unique(labels)):
        idx = labels == lab
        name = "Noise" if lab == -1 else f"C{lab}"

        plt.scatter(
            Z[idx, 0],
            Z[idx, 1],
            s=18,
            alpha=0.85,
            color=cmap(i % 10),
            label=name
        )

    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def add_cluster_counts(method_name, labels, cluster_counts):
    counts = pd.Series(labels).value_counts().sort_index()

    print(f"\n{method_name} cluster counts:")
    print(counts)

    for cid, cnt in counts.items():
        cluster_counts.append({
            "method": method_name,
            "cluster": int(cid),
            "count": int(cnt)
        })


def tune_dbscan_eps(X_dbscan):
    rows = []
    best = None
    total = len(X_dbscan)

    print("\nDBSCAN eps search:")

    for eps in DBSCAN_EPS_VALUES:
        labels = DBSCAN(
            eps=eps,
            min_samples=DBSCAN_MIN_SAMPLES
        ).fit_predict(X_dbscan)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_count = int(np.sum(labels == -1))
        noise_ratio = noise_count / total

        sil, dbi = safe_metrics(X_dbscan, labels)

        row = {
            "eps": eps,
            "num_clusters": n_clusters,
            "noise_count": noise_count,
            "noise_ratio": noise_ratio,
            "silhouette_score": sil,
            "davies_bouldin_index": dbi
        }

        rows.append(row)

        print(
            f"eps={eps} | clusters={n_clusters} | noise={noise_count} "
            f"| noise_ratio={noise_ratio:.3f} | sil={sil} | dbi={dbi}"
        )

        valid = n_clusters >= 2 and not np.isnan(sil) and noise_ratio <= 0.60

        if valid:
            if best is None or sil > best["silhouette_score"]:
                best = row.copy()
                best["labels"] = labels

    eps_df = pd.DataFrame(rows)

    if best is None:
        valid_rows = eps_df[
            (eps_df["num_clusters"] >= 2) &
            (~eps_df["silhouette_score"].isna())
        ]

        if len(valid_rows) > 0:
            best_row = valid_rows.sort_values(
                ["noise_ratio", "silhouette_score"],
                ascending=[True, False]
            ).iloc[0]

            eps = best_row["eps"]

            labels = DBSCAN(
                eps=eps,
                min_samples=DBSCAN_MIN_SAMPLES
            ).fit_predict(X_dbscan)

            best = best_row.to_dict()
            best["labels"] = labels

        else:
            eps = DBSCAN_EPS_VALUES[-1]

            labels = DBSCAN(
                eps=eps,
                min_samples=DBSCAN_MIN_SAMPLES
            ).fit_predict(X_dbscan)

            sil, dbi = safe_metrics(X_dbscan, labels)

            best = {
                "eps": eps,
                "num_clusters": len(set(labels)) - (1 if -1 in labels else 0),
                "noise_count": int(np.sum(labels == -1)),
                "noise_ratio": float(np.mean(labels == -1)),
                "silhouette_score": sil,
                "davies_bouldin_index": dbi,
                "labels": labels
            }

    return best, eps_df


def evaluate_medium():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    z_mean = np.load(os.path.join(RESULT_DIR, "z_mean_medium.npy"))
    metadata = pd.read_csv(os.path.join(SAVE_DIR, "metadata.csv"))

    X = z_mean

    Z2 = run_umap(X, random_state=RANDOM_SEED)

    results = []
    cluster_counts = []
    cluster_assignments = metadata.copy()

    labels_km = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_SEED,
        n_init=20
    ).fit_predict(X)

    sil, dbi = safe_metrics(X, labels_km)

    results.append({
        "method": "KMeans",
        "feature_space": "multimodal_vae_latent_audio+lyrics",
        "k": N_CLUSTERS,
        "silhouette_score": sil,
        "davies_bouldin_index": dbi,
        "num_clusters": len(set(labels_km))
    })

    cluster_assignments["kmeans"] = labels_km
    add_cluster_counts("KMeans", labels_km, cluster_counts)

    plot_embedding(
        Z2,
        labels_km,
        f"KMeans on Multimodal VAE Latent Space",
        os.path.join(PLOT_DIR, "kmeans_multimodal_vae.pdf")
    )

    labels_agg = AgglomerativeClustering(
        n_clusters=N_CLUSTERS
    ).fit_predict(X)

    sil, dbi = safe_metrics(X, labels_agg)

    results.append({
        "method": "Agglomerative",
        "feature_space": "multimodal_vae_latent_audio+lyrics",
        "k": N_CLUSTERS,
        "silhouette_score": sil,
        "davies_bouldin_index": dbi,
        "num_clusters": len(set(labels_agg))
    })

    cluster_assignments["agg"] = labels_agg
    add_cluster_counts("Agglomerative", labels_agg, cluster_counts)

    plot_embedding(
        Z2,
        labels_agg,
        "Agglomerative on Multimodal VAE Latent Space",
        os.path.join(PLOT_DIR, "agg_multimodal_vae.pdf")
    )

    X_scaled = StandardScaler().fit_transform(X)

    X_dbscan = PCA(
        n_components=min(DBSCAN_PCA_DIM, X_scaled.shape[1]),
        random_state=RANDOM_SEED
    ).fit_transform(X_scaled)

    best_db, eps_df = tune_dbscan_eps(X_dbscan)
    labels_db = best_db["labels"]

    results.append({
        "method": "DBSCAN",
        "feature_space": "PCA-reduced_multimodal_vae_latent_audio+lyrics",
        "k": f"auto_eps_{best_db['eps']}",
        "silhouette_score": best_db["silhouette_score"],
        "davies_bouldin_index": best_db["davies_bouldin_index"],
        "num_clusters": best_db["num_clusters"]
    })

    cluster_assignments["dbscan"] = labels_db
    add_cluster_counts(f"DBSCAN eps={best_db['eps']}", labels_db, cluster_counts)

    Z2_dbscan = run_umap(X_dbscan, random_state=RANDOM_SEED)

    plot_embedding(
        Z2_dbscan,
        labels_db,
        "DBSCAN on PCA-reduced Multimodal VAE Latent Space",
        os.path.join(PLOT_DIR, "dbscan_multimodal_vae.pdf")
    )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        os.path.join(RESULT_DIR, "medium_results.csv"),
        index=False
    )

    eps_df.to_csv(
        os.path.join(RESULT_DIR, "dbscan_eps_search.csv"),
        index=False
    )

    cluster_assignments.to_csv(
        os.path.join(RESULT_DIR, "medium_cluster_assignments.csv"),
        index=False
    )

    pd.DataFrame(cluster_counts).to_csv(
        os.path.join(RESULT_DIR, "medium_cluster_counts.csv"),
        index=False
    )

    print("\n===== MEDIUM RESULTS =====")
    print(results_df)