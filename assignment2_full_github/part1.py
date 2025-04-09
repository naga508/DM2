
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_circles, make_moons, make_blobs

# ----------------------------------------------------------------------
"""
Part 1:
Evaluation of k-Means over Diverse Datasets.
"""

def load_datasets(n_samples=100, random_state=42):
    datasets = {}
    X, y = make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=random_state)
    datasets["nc"] = (X, y)
    X, y = make_moons(n_samples=n_samples, noise=0.05, random_state=random_state)
    datasets["nm"] = (X, y)
    X, y = make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
    datasets["bvv"] = (X, y)
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    datasets["add"] = (X_aniso, y)
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    datasets["b"] = (X, y)
    return datasets

def fit_kmeans(dataset: tuple[np.ndarray, np.ndarray], *, k: int, seed: int = 42) -> np.ndarray:
    X, _ = dataset
    X_scaled = StandardScaler().fit_transform(X)
    kmeans = cluster.KMeans(n_clusters=k, init="random", n_init=10, random_state=seed)
    return kmeans.fit_predict(X_scaled)

def compute() -> dict:
    answers = {}
    data = load_datasets(n_samples=100)
    answers["1A: datasets"] = {k: (v[0].shape, v[1].shape) for k, v in data.items()}
    answers["1B: fit_kmeans"] = fit_kmeans

    ks = [2, 3, 5, 10]
    fig, axes = plt.subplots(len(ks), len(data), figsize=(20, 16))
    success = {}
    failure = []

    for col, (name, dataset) in enumerate(data.items()):
        for row, k in enumerate(ks):
            pred = fit_kmeans(dataset, k=k)
            axes[row, col].scatter(dataset[0][:, 0], dataset[0][:, 1], c=pred, cmap="tab10", s=10)
            axes[row, col].set_title(f"{name}, k={k}")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

        if name == "b":
            success[name] = {2, 3, 5, 10}
        elif name == "bvv":
            success[name] = {2, 3}
        elif name == "add":
            success[name] = {3}
        elif name == "nc":
            failure.append(name)
        elif name == "nm":
            failure.append(name)

    plt.tight_layout()
    plt.savefig("part1_clusters.pdf")
    answers["1C: cluster successes"] = success
    answers["1C: cluster failures"] = failure
    answers["1D: datasets sensitive to initialization"] = ["bvv", "add"]

    return answers

if __name__ == "__main__":
    answers = compute()
    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)
