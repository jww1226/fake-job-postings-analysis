import numpy as np
from scipy.sparse import issparse


def cosine_distance(v1, v2):
    """Compute the cosine distance between the two input vectors.

    Args:
        v1: A (sparse or dense) vector.
        v2: Another (sparse or dense) vector.

    Returns:
        float: The cosine distance between `v1` and `v2`.
    """
    # TODO: compute the cosine distance between two input vectors
    #       the implementation should work for both sparse and
    #       dense input vectors.

    v1 = v1.A.reshape(-1) if issparse(v1) else v1
    v2 = v2.A.reshape(-1) if issparse(v2) else v2
    v1_norm = np.sqrt(np.dot(v1, v1))
    v2_norm = np.sqrt(np.dot(v2, v2))
    cos_sim = np.dot(v1, v2) / (v1_norm * v2_norm)
    dist = 1. - cos_sim

    return dist


def compute_distances(data, centroids):
    """compute the cosine distances between every data point and
    every centroid.

    Args:
        data: A (sparse or dense) matrix of features for N documents.
            Each row represents a document.
        centroids (np.ndarray): The K cluster centres. Each row
            represent a cluster centre.

    Returns:
        np.ndarray: An N x K matrix of cosine distances.
    """
    # check the input
    assert data.shape[1] == centroids.shape[1]

    N = data.shape[0]
    K = centroids.shape[0]
    dists = np.full((N, K), -1.)

    # TODO: Compute the distances between data points and centroids
    #       such that dists[i, j] is the cosine distance between 
    #       the i-th data point and the j-th centroid.

    for n in range(N):
        for k in range(K):
            dists[n, k] = cosine_distance(data[n, :], centroids[k, :])

    return dists


def assign_data_points(distances):
    """Assign each data point to its closest centroid.

    Args:
        distances (np.ndarray): An N x K matrix where distances[i, j]
            is the cosine distance between the i-th data point and
            the j-th centroid.

    Returns:
        np.ndarray: A vector of size N.
    """
    N, K = distances.shape
    clusters = np.full(N, -1)

    # TODO: Assign each data point to its closest centroid such that
    #       clusters[i] = j denotes that the i-th data point is
    #       assigned to the j-th centroid.
    
    clusters[:] = np.argmin(distances, axis=1)

    return clusters


def update_centroids(data, centroids, clusters):
    """Re-compute each centroid as the average of the data points
    assigned to it.

    Args:
        data: A (sparse or dense) matrix of features for N documents.
            Each row represents a document.
        centroids (np.ndarray): The K cluster centres. Each row
            represent a cluster centre.
        clusters (np.ndarray): A vector of size N where clusters[i] = j
            denotes that the i-th data point is assigned to the j-th
            centroid.

    Returns:
        np.ndarray: The updated centroids.
    """
    # check the input
    assert data.shape[1] == centroids.shape[1]
    N = data.shape[0]
    K = centroids.shape[0]
    assert clusters.shape[0] == N

    # TODO: Re-compute each centroid as the average of the data points
    #       assigned to it.

    for k in range(K):
        ix = np.nonzero(clusters == k)[0]
        centroids[k, :] = data[ix].mean(axis=0)

    return centroids


def kmeans(data, K, max_iter=10, rng=None):
    """Clustering data points using the KMeans algorithm.

    Args:
        data: A matrix of features of documents. Each row represents a document.
        K (int): The number of cluster centres.
        max_iter (int): The maximum number of iterations to run in the KMeans algorithm.
        rng (np.random.Generator): A random number generator.

    Returns:
        centroids (np.ndarray): The cluster centres (after the re-computation of centroids).
        clusters (np.ndarray): The index of cluster each document belongs to, e.g., clusters[i] = k
            denotes that the i-th document is in the k-th cluster.
    """
    print(f'Clustering using KMeans (K={K}) ...')
    N = data.shape[0]
    assert N >= K
    rng = np.random.default_rng(rng)
    indices = rng.choice(N, size=K, replace=False)
    if issparse(data):
        centroids = data[indices, :].A  # dense
    else:
        centroids = data[indices, :]
    
    print(f'{"Iteration":>10} {"Total Distance":>20}')
    prev_clusters = None
    for i in range(max_iter):
        dists = compute_distances(data, centroids)
        clusters = assign_data_points(dists)
        centroids = update_centroids(data, centroids, clusters)
        print(f'{i:>10} {format(dists.min(axis=1).sum(), ".3f"):>20}')
        if prev_clusters is not None and np.all(prev_clusters == clusters):
            return centroids, clusters
        prev_clusters = clusters
    return centroids, clusters

