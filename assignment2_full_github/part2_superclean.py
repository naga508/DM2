
import pickle
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def fit_kmeans_sse(dataset: tuple[np.ndarray, np.ndarray, np.ndarray], *, k: int, seed: int = 42) -> float:
    X, _, _ = dataset
    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=k, init="random", n_init=10, random_state=seed)
    labels = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_
    return sum(np.sum((X_scaled[labels == i] - centroids[i])**2) for i in range(k))

def fit_kmeans_inertia(dataset: tuple[np.ndarray, np.ndarray, np.ndarray], *, k: int, seed: int = 42) -> float:
    X, _, _ = dataset
    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=k, init="random", n_init=10, random_state=seed)
    kmeans.fit(X_scaled)
    return kmeans.inertia_

def compute():
    answers = {}
    X, y, centers = make_blobs(center_box=(-20, 20), n_samples=20, centers=5, random_state=12, return_centers=True)
    answers["2A: blob"] = [np.array(X), np.array(y), np.array(centers)]
    answers["2B: fit_kmeans"] = fit_kmeans_sse
    answers["2C: SSE plot"] = [[float(k), round(float(fit_kmeans_sse((X, y, centers), k=k)), 5)] for k in range(1, 9)]
    answers["2D: inertia plot"] = [[float(k), round(float(fit_kmeans_inertia((X, y, centers), k=k)), 5)] for k in range(1, 9)]
    answers["2D: do ks agree?"] = "yes"
    return answers

if __name__ == "__main__":
    with open("part2.pkl", "wb") as f:
        pickle.dump(compute(), f)
