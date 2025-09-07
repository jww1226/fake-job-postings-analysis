import nltk
import string
import numpy as np
from tqdm import tqdm
from gensim.models.keyedvectors import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer


tokeniser = nltk.tokenize.TreebankWordTokenizer()
stopwords = frozenset(nltk.corpus.stopwords.words("english"))
trans_table = str.maketrans(dict.fromkeys(string.punctuation))

def tokenise_text(str_):
    """Tokenize a string of text.

    Args:
        str_: The input string of text.

    Returns:
        list(str): A list of tokens.
    """
    # for simplicity, remove non-ASCII characters
    str_ = str_.encode(encoding='ascii', errors='ignore').decode()
    return [t for t in tokeniser.tokenize(str_.lower().translate(trans_table)) if t not in stopwords]


def get_features_tfidf(Xr_fit, Xr_pred=None):
    """Given the training documents, each represented as a string,
    return a sparse matrix of TF-IDF features.

    Args:
        Xr_fit (iterable(str)): The input documents, each represented
            as a string.
        Xr_pred (iterable(str)): Optional input documents, each 
            represented as a string. Documents in Xr_pred should NOT
            be used to compute the IDF (which should be computed using
            documents in Xr_fit).
    Returns:
        X_fit: A sparse matrix of TF-IDF features of documents in Xr_fit.
        X_pred: Optional. A sparse matrix of TF-IDF features of documents
            in Xr_pred if it is provided.
    """
    # TODO: compute the TF-IDF features of the input documents.
    #   You may want to use TfidfVectorizer in the scikit-learn package,
    #   see https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    print('Generating features (TF-IDF) ...')

    text_analyser = tokenise_text
    vectoriser = TfidfVectorizer(analyzer=text_analyser)
    X_fit = vectoriser.fit_transform(Xr_fit)
    if Xr_pred is not None:
        X_pred = vectoriser.transform(Xr_pred)

    return X_fit if Xr_pred is None else (X_fit, X_pred)


def document_to_vector(tokenised_doc, word_vectors):
    """Takes a (tokenised) document and turns it into a vector by aggregating
    its word vectors.

    Args:
        tokenised_doc (list(list(str))): A document represented as list of
            sentences. Each sentence is a list of tokens.
        word_vectors (gensim.models.keyedvectors.KeyedVectors): A mapping 
            from words (string) to their embeddings (np.ndarray)

    Returns:
        np.array: The aggregated word vector representing the input document.
    """
    # check the input
    assert isinstance(word_vectors, KeyedVectors)
    vector_size = word_vectors.vector_size

    # TODO: convert each document into a vector
    aggregate = 'mean'
    assert aggregate in ('mean', 'max', 'sum')
    func_dict = {'mean': np.mean, 'max': np.max, 'sum': np.sum}
    vecs = [word_vectors[t] for s in tokenised_doc for t in s if t in word_vectors]
    if len(vecs) == 0:
        vec = np.zeros(vector_size)
    else:
        func = func_dict[aggregate]
        vec = func(vecs, axis=0)
    return vec


def get_features_w2v(Xt, word_vectors):
    """Given a dataset of (tokenised) documents (each represented as a list of
    tokenised sentences), return a (dense) matrix of aggregated word vector for
    each document in the dataset.

    Args:
        Xt (list(list(list(str)))): A list of (tokenised) documents (each
            represented as a list of tokenised sentences where a sentence is a
            list of tokens).
        word_vectors (gensim.models.keyedvectors.KeyedVectors): A mapping 
            from words (string) to their embeddings (np.ndarray)

    Returns:
        np.ndarray: A matrix of features. The i-th row vector represents the i-th
            document in `Xr`.
    """
    print('Generating features (word2vec) ...')
    return np.vstack([document_to_vector(xt, word_vectors) for xt in tqdm(Xt)])

