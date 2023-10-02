import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gensim as gs
import gensim.downloader as gs_api
import spacy
import string
from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from LogisticRegression import LogisticRegression as Lr
from sklearn.linear_model import LogisticRegression as Lr_skl


# Politeness Classifier

def read_politeness_data():
    data = pd.read_csv('datasets/politeness_data.csv')
    data.info()
    docs_array = data['text'].to_numpy()
    Y_array = data['polite'].to_numpy()
    return docs_array, Y_array

def simple_data_split(docs_array, Y_array, length, split_mark=0.7):
    if split_mark > 0. and split_mark < 1.0:
        n = int(split_mark * length)
    else :
        n = int(split_mark)
    X_train = docs_array[:n]
    X_test = docs_array[n:]
    Y_train = Y_array[:n]
    Y_test = Y_array[n:]
    return X_train, X_test, Y_train, Y_test

def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)

def custom_logistic_regression_with_tdm(X_train_vec, X_test_vec):
    print('-----\nPerforming Custom Logistic Regression Class')
    X_train_tdm = X_train_vec.toarray()
    X_test_tdm = X_test_vec.toarray()
    # use logistic regression
    lr = Lr(lr=2.125, n_iters=200)
    lr.fit(X_train_tdm, Y_train)
    y_pred = lr.predict(X_test_tdm)
    acc = accuracy(y_pred, Y_test)
    print('accuracy = ', acc)

def sklearn_logistic_regression_with_vector(X_train_vec, X_test_vec):
    print('-----\nPerforming Sklearn Logistic Regression with CountVectorizer vectors')
    # cross-validation score
    scores = cross_val_score(Lr_skl(), X_train_vec, Y_train, cv=5)
    print('cross-validation score =', np.mean(scores))
    # mean accuracy score
    lr = Lr_skl()
    lr.fit(X_train_vec, Y_train)
    print('training set score =', lr.score(X_train_vec, Y_train))
    print('test set score =', lr.score(X_test_vec, Y_test))
    lr_pred = lr.predict(X_test_vec)
    cm = confusion_matrix(Y_test, lr_pred)
    print('confusion matrix =', cm)  

# read data from CSV
docs_array, Y_array = read_politeness_data()

# split raw data into training/testing 
X_train, X_test, Y_train, Y_test = simple_data_split(
    docs_array,
    Y_array,
    len(docs_array),
)

# use vectorizer
vector = CountVectorizer()
X_train_vec = vector.fit_transform(X_train)
X_test_vec = vector.transform(X_test)

# custom_logistic_regression_with_tdm(X_train_vec, X_test_vec)

# sklearn_logistic_regression_with_vector(X_train_vec, X_test_vec)

# gensim for pre-trained word vectors
# ['fasttext-wiki-news-subwords-300', 'conceptnet-numberbatch-17-06-300', 'word2vec-ruscorpora-300', 'word2vec-google-news-300', 'glove-wiki-gigaword-50', 'glove-wiki-gigaword-100', 'glove-wiki-gigaword-200', 'glove-wiki-gigaword-300', 'glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 'glove-twitter-200', '__testing_word2vec-matrix-synopsis']
# ptwv = gs_api.load('glove-twitter-50')

