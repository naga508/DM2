
import pickle
from typing import Literal, Any
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage as linkage_fct
from scipy.signal import savgol_filter
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

def fit_hierarchical_cluster(
    dataset: tuple[NDArray],
    linkage: Literal["ward", "average", "complete", "single"],
    k: int,
) -> NDArray:
    X = dataset[0]
    X_scaled = StandardScaler().fit_transform(X)
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    return model.fit_predict(X_scaled)

def get_distance_threshold(Z) -> tuple[dict[str, Any], dict[str, Any]]:
    distances = Z[:, 2]
    n = len(distances)
    smooth_dists = savgol_filter(distances, window_length=5 if n >= 5 else 3, polyorder=2)
    slope = np.gradient(smooth_dists)
    curvature = np.gradient(slope)
    max_slope_idx = int(np.argmax(slope))
    max_curvature_idx = int(np.argmax(np.abs(curvature)))
    threshold_slope = smooth_dists[max_slope_idx]
    threshold_curvature = smooth_dists[max_curvature_idx]
    return (
        {"threshold": threshold_slope, "index": max_slope_idx},
        {"threshold": threshold_curvature, "index": max_curvature_idx},
    )

def fit_modified(
    dataset: tuple[NDArray],
    linkage: Literal["ward", "average", "complete", "single"],
) -> tuple[NDArray, NDArray, dict, dict]:
    X = dataset[0]
    X_scaled = StandardScaler().fit_transform(X)
    Z = linkage_fct(X_scaled, method=linkage)
    slope_info, curvature_info = get_distance_threshold(Z)
    labels_slope = fcluster(Z, t=slope_info["threshold"], criterion="distance")
    labels_curvature = fcluster(Z, t=curvature_info["threshold"], criterion="distance")
    return labels_slope, labels_curvature, slope_info, curvature_info

def load_datasets(n_samples=100, random_state=42):
    from sklearn.datasets import make_circles, make_moons, make_blobs
    datasets = {}
    X, _ = make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=random_state)
    datasets["nc"] = (X,)
    X, _ = make_moons(n_samples=n_samples, noise=0.05, random_state=random_state)
    datasets["nm"] = (X,)
    X, _ = make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
    datasets["bvv"] = (X,)
    X, _ = make_blobs(n_samples=n_samples, random_state=random_state)
    X_aniso = np.dot(X, [[0.6, -0.6], [-0.4, 0.8]])
    datasets["add"] = (X_aniso,)
    X, _ = make_blobs(n_samples=n_samples, random_state=random_state)
    datasets["b"] = (X,)
    return datasets

def compute():
    answers = {}
    data = load_datasets(n_samples=100)
    answers["4A: datasets"] = {k: v[0].shape for k, v in data.items()}
    answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster

    linkages = ["single", "complete", "ward", "average"]
    fig, axes = plt.subplots(len(linkages), len(data), figsize=(20, 16))
    successful_datasets = []

    for row, linkage in enumerate(linkages):
        for col, (name, dataset) in enumerate(data.items()):
            labels = fit_hierarchical_cluster(dataset, linkage=linkage, k=2)
            axes[row, col].scatter(dataset[0][:, 0], dataset[0][:, 1], c=labels, cmap="tab10", s=10)
            axes[row, col].set_title(f"{name}, {linkage}")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        if linkage == "complete":
            successful_datasets.append("nc")
        if linkage == "average":
            successful_datasets.append("nm")

    plt.tight_layout()
    plt.savefig("part4_hierarchical_linkages.pdf")
    answers["4B: cluster successes"] = sorted(set(successful_datasets))

    fig, axes = plt.subplots(2, len(data), figsize=(20, 8))
    for col, (name, dataset) in enumerate(data.items()):
        labels_slope, labels_curvature, _, _ = fit_modified(dataset, "ward")
        axes[0, col].scatter(dataset[0][:, 0], dataset[0][:, 1], c=labels_slope, cmap="tab10", s=10)
        axes[1, col].scatter(dataset[0][:, 0], dataset[0][:, 1], c=labels_curvature, cmap="tab10", s=10)
        axes[0, col].set_title(f"{name} (slope)")
        axes[1, col].set_title(f"{name} (curvature)")
        axes[0, col].set_xticks([]); axes[1, col].set_xticks([])
        axes[0, col].set_yticks([]); axes[1, col].set_yticks([])

    plt.tight_layout()
    plt.savefig("part4_thresholds.pdf")
    answers["4C: modified function"] = fit_modified

    return answers

if __name__ == "__main__":
    answers = compute()
    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)
