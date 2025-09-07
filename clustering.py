import os
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import issparse
from sklearn.manifold import TSNE
from sklearn.decomposition import (
    PCA,
    TruncatedSVD,
)
from word2vec import train_w2v
from kmeans import (
    kmeans,
    cosine_distance,
)
from features import (
    tokenise_text,
    get_features_tfidf,
    get_features_w2v,
)


def visualise_clusters(X, clusters, title=None, fout=None):
    """Visualising the clustering.

    Args:
        X: A matrix of features of documents. Each row represents a document.
        clusters (np.ndarray): The index of cluster each document belongs to, e.g., clusters[i] = k
            denotes that the i-th document is in the k-th cluster.
        title (str): Optional. The title of the figure.
        fout (str): Optional. The output figure filename.
    """
    print('Reducing dimensionality ...')
    # dim = 50
    dim = 20
    if isinstance(X, np.ndarray):
        Z = PCA(n_components=dim).fit_transform(X) if X.shape[1] > dim else X
    elif issparse(X):
        if X.shape[1] > dim:
            Z = TruncatedSVD(n_components=dim, n_iter=10, random_state=101).fit_transform(X)
        else:
            Z = X.A
    else:
        assert False
    # dist_metric = cosine_distance
    dist_metric = 'cosine'
    Z = TSNE(metric=dist_metric, init='pca', random_state=101).fit_transform(Z)
    print('Visualising clusters ...')
    points_colour = clusters
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(Z[:, 0], Z[:, 1], c=points_colour, s=5, alpha=0.9)
    # remove ticks and tick labels
    plt.tick_params(bottom=False, labelbottom=False,
                    top=False, labeltop=False,
                    left=False, labelleft=False,
                    right=False, labelright=False)
    plt.title(title, size=14)
    if fout is not None:
        plt.savefig(fout, bbox_inches='tight')
    plt.show()


def cluster_reviews_tfidf(Xr, K, max_iter, rng=None):
    """Clustering documents represented by TF-IDF features.

    Args:
        Xr (iterable(str): Documents to be clustered, each represented as a string.
        K (int): The number of clusters.
        max_iter (int): The maximum number of iterations to run in the KMeans algorithm.
        rng (np.random.Generator): A random number generator.
    """
    assert K < len(Xr)
    rng = np.random.default_rng(rng)
    X = get_features_tfidf(Xr)
    new_centroids, clusters = kmeans(X, K, max_iter=max_iter, rng=rng)
    title = f'KMeans Clustering (K: {K}, Features: TF-IDF)'
    output_file = f'clusters_tfidf_K{K}.png'
    visualise_clusters(X, clusters, title=title, fout=output_file)
    

def cluster_reviews_w2v(Xr, K, max_iter, rng=None):
    """Clustering documents represented by aggregated word vectors.

    Args:
        Xr (iterable(str): Documents to be clustered, each represented as a string.
        K (int): The number of clusters.
        max_iter (int): The maximum number of iterations to run the KMeans algorithm.
        rng (np.random.Generator): A random number generator.
    """

    # TODO: perform functionality similar to that of `cluster_reviews_tfidf` but use
    #       aggregated Word2Vec word vectors instead of TF-IDF features.
    #       You may reuse the best hyper-parameters found in Part B of Question 1 
    #       for training the Word2Vec model.

    assert K < len(Xr)
    rng = np.random.default_rng(rng)
    get_sentences = lambda text: [tokenise_text(sent) for sent in nltk.tokenize.sent_tokenize(text)]
    Xt = [get_sentences(xr) for xr in tqdm(Xr)]

    # could reuse word2vec hyper-parameters found in Part B of Q1
    w2v_params = {'vector_size': 300, 'epochs': 5, 'negative': 10}
    word_vectors = train_w2v(Xt, **w2v_params)

    X = get_features_w2v(Xt, word_vectors)
    new_centroids, clusters = kmeans(X, K, max_iter=max_iter, rng=rng)
    title = f'KMeans Clustering (K: {K}, Features: word2vec)'
    output_file = f'clusters_word2vec_K{K}.png'
    visualise_clusters(X, clusters, title=title, fout=output_file)


if __name__ == '__main__':
    data_file = os.path.join("data", "movie_reviews_unlabelled.csv")
    print('Loading data ...')
    df = pd.read_csv(data_file)
    Xr = df["text"].tolist()

    K = 5
    max_iter = 20
    seed = 101
    rng = np.random.default_rng(seed=seed)
    cluster_reviews_tfidf(Xr, K, max_iter, rng)
    # cluster_reviews_w2v(Xr, K, max_iter, rng)

