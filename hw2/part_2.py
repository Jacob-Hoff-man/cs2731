import numpy as np
import pandas as pd
import gensim.downloader as gs_api
from sklearn.neural_network import MLPClassifier
import spacy
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from logistic_regression import LogisticRegression as Lr
from sklearn.linear_model import LogisticRegression as Lr_skl
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Globals
print('Loading globals (spacy(en_core_web_sm) and gensim(fasttext-wiki-news-subwords-300))')
nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words
punctuations = string.punctuation
# ['fasttext-wiki-news-subwords-300', 'conceptnet-numberbatch-17-06-300', 'word2vec-ruscorpora-300', 'word2vec-google-news-300', 'glove-wiki-gigaword-50', 'glove-wiki-gigaword-100', 'glove-wiki-gigaword-200', 'glove-wiki-gigaword-300', 'glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 'glove-twitter-200', '__testing_word2vec-matrix-synopsis']
ptwv = gs_api.load('fasttext-wiki-news-subwords-300')
print('Globals loaded')

# Politeness Classifier

def read_data(input_file_name):
    data = pd.read_csv('datasets/' + input_file_name)
    docs_array = data['text'].to_numpy()
    Y_array = data['polite'].to_numpy()
    return docs_array, Y_array

def read_politeness_data_raw(input_file_name):
    return pd.read_csv('datasets/' + input_file_name)

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
    return np.sum(y_pred == y_test) / len(y_test)

def custom_logistic_regression_with_tdm(X_train_vec, X_test_vec, Y_train, Y_test):
    print('-----\nPerforming Custom Logistic Regression Class')
    X_train_tdm = X_train_vec.toarray()
    X_test_tdm = X_test_vec.toarray()
    # use logistic regression
    lr = Lr(lr=2.125, n_iters=200)
    lr.fit(X_train_tdm, Y_train)
    y_pred = lr.predict(X_test_tdm)
    acc = accuracy(y_pred, Y_test)
    print('accuracy = ', acc)

def sklearn_logistic_regression_with_vector(X_train_vec, X_test_vec, Y_train, Y_test):
    print('-----\nPerforming Sklearn Logistic Regression with CountVectorizer vectors')
    # cross-validation score
    scores = cross_val_score(Lr_skl(), X_train_vec, Y_train, cv=5)
    # mean accuracy score
    lr = Lr_skl()
    lr.fit(X_train_vec, Y_train)
    predicted = lr.predict(X_test_vec)
    print_metrics(scores, predicted, Y_test)

def sklearn_logistic_regression_with_static_word_embedding_and_pre_processing(X_train, X_test, Y_train, Y_test, input_test_file_name):
    classifier = Lr_skl()
    classifier.fit(X_train, Y_train)
    predicted = classifier.predict(X_test)
    scores = cross_val_score(Lr_skl(), X_train, Y_train, cv=5)
    print_metrics(scores, predicted, Y_test)
    if input_test_file_name:
        print('Performing trained classifier on held-out test dataset: ', input_test_file_name)
        held_out_pre_processed_data = pre_process_data(input_test_file_name)
        held_out_X = held_out_pre_processed_data['vec'].to_list()
        held_out_Y = held_out_pre_processed_data['polite'].to_list()
        predicted = classifier.predict(held_out_X)
        scores = cross_val_score(classifier, held_out_X, held_out_Y, cv=5)
        print_metrics(scores, predicted, held_out_Y)

def spacy_tokenizer(sentence, remove_stop_words=False):
    doc = nlp(sentence)
    tokens = [ word.norm_.lower().strip() for word in doc ]
    if remove_stop_words:
        tokens = [ word for word in tokens if word not in stop_words and word not in punctuations ]
    return tokens

def sent_vec(sent):
    vector_size = ptwv.vector_size
    wv_res = np.zeros(vector_size)
    # print(wv_res)
    ctr = 1
    for w in sent:
        if w in ptwv:
            ctr += 1
            wv_res += ptwv[w]
    wv_res = wv_res / ctr
    return wv_res

def pre_process_data(input_file_name):
    data = read_politeness_data_raw(input_file_name)
    # pre-processing
    data['tokens'] = data['text'].apply(spacy_tokenizer)
    # static word embedding vector
    data['vec'] = data['tokens'].apply(sent_vec)
    return data

def print_metrics(scores, predicted, Y_test):
    print('5-fold Cross-validation score:', np.mean(scores))
    print("Accuracy:",metrics.accuracy_score(Y_test, predicted))
    print("Precision:",metrics.precision_score(Y_test, predicted))
    print("Recall:",metrics.recall_score(Y_test, predicted))
    print("F1 Score:",metrics.f1_score(Y_test, predicted))

def perform_sklearn_lr_with_tdm(input_file_name):
    print('Performing Sklearn LR with tdm')
    docs_array, Y_array = read_data(input_file_name)
    X_train, X_test, Y_train, Y_test = simple_data_split(docs_array, Y_array, len(docs_array))
    # use vectorizer
    vector = CountVectorizer()
    X_train_vec = vector.fit_transform(X_train)
    X_test_vec = vector.transform(X_test)
    sklearn_logistic_regression_with_vector(X_train_vec, X_test_vec, Y_train, Y_test)

def perform_sklearn_lr_with_static_word_embeddings_and_pre_processing(input_file_name, input_test_file_name):
    print('Performing Sklearn LR with static word embeddings and pre-processing')
    pre_processed_data = pre_process_data(input_file_name)
    X = pre_processed_data['vec'].to_list()
    Y = pre_processed_data['polite'].to_list()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,stratify=Y)
    sklearn_logistic_regression_with_static_word_embedding_and_pre_processing(X_train, X_test, Y_train, Y_test, input_test_file_name)
    
def perform_sklearn_backpropagation_neural_network_with_static_word_embeddings_and_pre_processeing(input_file_name, input_test_file_name):
    print('Performing Sklearn MLPClassifier with static word embeddings and pre-processing')
    pre_processed_data = pre_process_data(input_file_name)
    X = pre_processed_data['vec'].to_list()
    Y = pre_processed_data['polite'].to_list()
    print('     Pre-processing data completed')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,stratify=Y)
    mlp_classifier = MLPClassifier(
        hidden_layer_sizes=(150,100,50),
        max_iter = 300,
        activation = 'relu',
        solver = 'adam')
    mlp_classifier.fit(X_train, Y_train)
    print('     Model fit completed')
    predicted = mlp_classifier.predict(X_test)
    scores = cross_val_score(mlp_classifier, X_train, Y_train, cv=5)
    print_metrics(scores, predicted, Y_test)
    if input_test_file_name:
        print('Performing trained classifier on held-out test dataset: ', input_test_file_name)
        held_out_pre_processed_data = pre_process_data(input_test_file_name)
        held_out_X = held_out_pre_processed_data['vec'].to_list()
        held_out_Y = held_out_pre_processed_data['polite'].to_list()
        predicted = mlp_classifier.predict(held_out_X)
        scores = cross_val_score(mlp_classifier, held_out_X, held_out_Y, cv=5)
        print_metrics(scores, predicted, held_out_Y)

# perform_sklearn_lr_with_tdm()
# perform_sklearn_lr_with_static_word_embeddings_and_pre_processing('politeness_data.csv')
# perform_sklearn_backpropagation_neural_network_with_static_word_embeddings_and_pre_processeing('politeness_data.csv')


