
import pickle
import numpy as np
from scipy.io import loadmat
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import euclidean

def single_link_fn(X, cluster_I, cluster_J):
    return min(euclidean(X[i], X[j]) for i in cluster_I for j in cluster_J)

def compute():
    answers = {}
    data = loadmat("hierarchical_toy_data.mat")
    X = np.array(data["X"])
    answers["3A: toy data"] = {"X": X}
    Z = linkage(X, method="single")
    dendro_data = dendrogram(Z, no_plot=True)
    answers["3B: linkage"] = Z
    answers["3B: dendogram"] = dendro_data

    I, J = set([8, 2, 13]), set([1, 9])
    clusters = {i: [i] for i in range(len(X))}
    merge_iteration = None
    for idx, (a, b, _, _) in enumerate(Z):
        a, b = int(a), int(b)
        new_cluster = clusters[a] + clusters[b]
        clusters[len(X) + idx] = new_cluster
        if I.issubset(new_cluster) and any(set(c) == J for c in [clusters[a], clusters[b]]):
            merge_iteration = idx
            break
    answers["3C: iteration"] = int(merge_iteration)
    answers["3D: function"] = single_link_fn
    answers["3D: min_dist"] = round(float(single_link_fn(X, list(I), list(J))), 5)

    current_clusters = {i: [i] for i in range(len(X))}
    for idx, (a, b, _, _) in enumerate(Z):
        if idx == merge_iteration:
            cluster_state = list(current_clusters.values())
            break
        a, b = int(a), int(b)
        current_clusters[len(X) + idx] = current_clusters[a] + current_clusters[b]
        del current_clusters[a]
        del current_clusters[b]

    answers["3E: clusters"] = {frozenset(map(int, c)) for c in cluster_state}
    answers["3F: rich get richer"] = (
        "Yes. The dendrogram shows chaining behavior where one cluster continues "
        "absorbing others, which reflects the 'rich get richer' phenomenon."
    )
    return answers

if __name__ == "__main__":
    with open("part3.pkl", "wb") as f:
        pickle.dump(compute(), f)
