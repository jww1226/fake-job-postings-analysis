import os
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from features import (
    tokenise_text,
    get_features_tfidf,
    get_features_w2v,
)
from word2vec import (
    train_w2v,
    search_hyperparams,
)
from classifier import (
    train_model,
    eval_model,
    search_C,
)


def prepare_dataset(filename):
    """Prepare the training/validation/test dataset.

    Args:
        filename (str): The name of file from which data will be loaded.

    Returns:
        Xr_train (iterable(str)): Documents in the training set, each 
            represented as a string.
        y_train (np.ndarray): A vector of class labels for documents in 
            the training set, each element of the vector is either 0 or 1.
        Xr_val (iterable(str)): Documents in the validation set, each 
            represented as a string.
        y_val (np.ndarray): A vector of class labels for documents in 
            the validation set, each element of the vector is either 0 or 1.
        Xr_test (iterable(str)): Documents in the test set, each 
            represented as a string.
        y_test (np.ndarray): A vector of class labels for documents in 
            the test set, each element of the vector is either 0 or 1.
    """
    print('Preparing train/val/test dataset ...')
    # load raw data
    df = pd.read_csv(filename)
    # shuffle the rows
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    # get the train, val, test splits
    train_frac, val_frac, test_frac = 0.7, 0.1, 0.2
    Xr = df["text"].tolist()
    train_end = int(train_frac*len(Xr))
    val_end = int((train_frac + val_frac)*len(Xr))
    Xr_train = Xr[0:train_end]
    Xr_val = Xr[train_end:val_end]
    Xr_test = Xr[val_end:]

    # encode sentiment labels ('pos' and 'neg')
    yr = df["label"].tolist()
    le = LabelEncoder()
    y = le.fit_transform(yr)
    y_train = np.array(y[0:train_end])
    y_val = np.array(y[train_end:val_end])
    y_test = np.array(y[val_end:])
    return Xr_train, y_train, Xr_val, y_val, Xr_test, y_test


def analyse_sentiment_tfidf(Xr_train, y_train, Xr_val, y_val, Xr_test, y_test):
    """Analyse sentiment using TF-IDF features.

    Args:
        Xr_train (iterable(str)): Documents in the training set, each 
            represented as a string.
        y_train (np.ndarray): A vector of class labels for documents in 
            the training set, each element of the vector is either 0 or 1.
        Xr_val (iterable(str)): Documents in the validation set, each 
            represented as a string.
        y_val (np.ndarray): A vector of class labels for documents in 
            the validation set, each element of the vector is either 0 or 1.
        Xr_test (iterable(str)): Documents in the test set, each 
            represented as a string.
        y_test (np.ndarray): A vector of class labels for documents in 
            the test set, each element of the vector is either 0 or 1.

    Returns:
        float: The accuracy of the sentiment classifier on the test set.
    """
    # generate TF-IDF features for texts in the training and validation sets 
    X_train, X_val = get_features_tfidf(Xr_train, Xr_val)

    # search for the best C value
    C = search_C(X_train, y_train, X_val, y_val)

    print('Analysing sentiment (TF-IDF) ...')

    # re-train the classifier using the training set concatenated with the
    # validation set and the best C value
    X_train_val, X_test = get_features_tfidf(Xr_train + Xr_val, Xr_test)
    y_train_val = np.concatenate([y_train, y_val], axis=-1)
    model = train_model(X_train_val, y_train_val, C)

    # evaluate performance on the test set
    acc = eval_model(X_test, y_test, model)
    return acc


def analyse_sentiment_w2v(Xr_train, y_train, Xr_val, y_val, Xr_test, y_test):
    """Analyse sentiment using aggregated word2vec word vectors.

    Args:
        Xr_train (iterable(str)): Documents in the training set, each 
            represented as a string.
        y_train (np.ndarray): A vector of class labels for documents in 
            the training set, each element of the vector is either 0 or 1.
        Xr_val (iterable(str)): Documents in the validation set, each 
            represented as a string.
        y_val (np.ndarray): A vector of class labels for documents in 
            the validation set, each element of the vector is either 0 or 1.
        Xr_test (iterable(str)): Documents in the test set, each 
            represented as a string.
        y_test (np.ndarray): A vector of class labels for documents in 
            the test set, each element of the vector is either 0 or 1.
        word2vec_model (Word2VecModel): A trained word2vec model.

    Returns:
        float: The accuracy of the sentiment classifier on the test set.
    """
    # prepare data
    get_sentences = lambda text: [tokenise_text(sent) for sent in nltk.tokenize.sent_tokenize(text)]
    Xt_train = [get_sentences(xr) for xr in tqdm(Xr_train)]
    Xt_val = [get_sentences(xr) for xr in tqdm(Xr_val)]
    Xt_test = [get_sentences(xr) for xr in tqdm(Xr_test)]

    # tune hyper-parameters
    best_params = search_hyperparams(Xt_train, y_train, Xt_val, y_val)
    assert 'C' in best_params
    best_C = best_params['C']
    del best_params['C']
    word_vectors = train_w2v(Xt_train + Xt_val, **best_params)
    
    # re-train the classifier using the training set concatenated with the
    # validation set 
    X_train_val = get_features_w2v(Xt_train + Xt_val, word_vectors)
    y_train_val = np.concatenate([y_train, y_val], axis=-1)
    X_test = get_features_w2v(Xt_test, word_vectors)
    print('Analysing sentiment (word2vec) ...')
    model = train_model(X_train_val, y_train_val, best_C)

    # evaluate performance on the test set
    acc = eval_model(X_test, y_test, model)
    return acc


if __name__ == '__main__':
    # split dataset into training, validation and test sets
    data_file = os.path.join("data", "movie_reviews_labelled.csv")
    Xr_train, y_train, Xr_val, y_val, Xr_test, y_test = prepare_dataset(filename=data_file)

    # uncomment to predict sentiment on the test set using TF-IDF features
    acc = analyse_sentiment_tfidf(Xr_train, y_train, Xr_val, y_val, Xr_test, y_test)
    print(f'Accuracy on test set (TF-IDF): {acc:.4f}')

    # uncomment to predict sentiment on the test set using aggregated word vectors
    # acc = analyse_sentiment_w2v(Xr_train, y_train, Xr_val, y_val, Xr_test, y_test)
    # print(f'Accuracy on test set (word2vec): {acc:.4f}')
    
