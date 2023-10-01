import os
import subprocess
import csv
import re
import random
import numpy as np
import scipy
import pandas as pd
import math
import csv

def read_in_shakespeare():
    """Reads in the Shakespeare dataset and processes it into a list of tuples.
       Also reads in the vocab and play name lists from files.

    Each tuple consists of
    tuple[0]: The name of the play
    tuple[1] A line from the play as a list of tokenized words.

    Returns:
      tuples: A list of tuples in the above format.
      document_names: A list of the plays present in the corpus.
      vocab: A list of all tokens in the vocabulary.
    """

    tuples = []

    with open('datasets/shakespeare_plays.csv') as f:
        csv_reader = csv.reader(f, delimiter=";")
        for row in csv_reader:
            play_name = row[1]
            line = row[5]
            line_tokens = re.sub(r"[^a-zA-Z0-9\s]", " ", line).split()
            line_tokens = [token.lower() for token in line_tokens]

            tuples.append((play_name, line_tokens))

    with open("vocab.txt") as f:
        vocab = [line.strip() for line in f]

    with open("play_names.txt") as f:
        document_names = [line.strip() for line in f]

    return tuples, document_names, vocab

def read_in_snli():
    """Reads in the SNLI and processes it into a list of tuples.
       Also reads in the vocab and sentence names from files.

    Each tuple consists of
    tuple[0]: The name of the sentence (sentenceId)
    tuple[1] the sentence as a list of tokenized words.

    Returns:
      tuples: A list of tuples in the above format.
    """

    tuples = []

    with open('datasets/snli.csv') as f:
        csv_reader = csv.reader(f, delimiter=",")
        for row in csv_reader:
            sentenceName = row[0]
            line = row[2]
            line_tokens = re.sub(r"[^a-zA-Z0-9\s]", " ", line).split()
            line_tokens = [token.lower() for token in line_tokens]

            tuples.append((sentenceName, line_tokens))

    return tuples

def get_row_vector(matrix, row_id):
    """A convenience function to get a particular row vector from a numpy matrix

    Inputs:
      matrix: a 2-dimensional numpy array
      row_id: an integer row_index for the desired row vector

    Returns:
      1-dimensional numpy array of the row vector
    """
    return matrix[row_id, :]

def get_column_vector(matrix, col_id):
    """A convenience function to get a particular column vector from a numpy matrix

    Inputs:
      matrix: a 2-dimensional numpy array
      col_id: an integer col_index for the desired row vector

    Returns:
      1-dimensional numpy array of the column vector
    """
    return matrix[:, col_id]

def get_doc_tuples(line_tuples):
  doc_tuples = {}
  for tuple in line_tuples:
    doc_tuples.setdefault(tuple[0],[]).append(tuple)
  return doc_tuples

def get_tokens_from_doc_tuples(doc_tuples):
  tokens = []
  for tuple in doc_tuples:
    for word in tuple[1]:
      tokens.append(word)
  return tokens

def get_unique_tokens(tokens):
  return list(set(tokens))

def get_doc_word_counts(tokens):
  word_counts = {}
  for word in tokens:
    if word in word_counts.keys():
      word_counts[word] += 1
    else:
      word_counts.setdefault(word, 1)
  return word_counts       

def create_term_document_matrix(line_tuples, document_names, vocab):
    """Returns a numpy array containing the term document matrix for the input lines.

    Inputs:
      line_tuples: A list of tuples, containing the name of the document and
      a tokenized line from that document.
      document_names: A list of the document names
      vocab: A list of the tokens in the vocabulary

    Let m = len(vocab) and n = len(document_names).

    Returns:
      td_matrix: A mxn numpy array where the number of rows is the number of words
          and each column corresponds to a document. A_ij contains the
          frequency with which word i occurs in document j.
    """
    m = len(vocab)
    n = len(document_names)
    doc_tuples = get_doc_tuples(line_tuples)
    term_document_matrix = np.zeros(shape=(m, n))
    for col_index, name in zip(range(n), document_names):
      tokens = get_tokens_from_doc_tuples(doc_tuples[name])
      doc_word_counts = get_doc_word_counts(tokens)
      for key in doc_word_counts.keys():
        matrix_row_index = vocab.index(key)
        matrix_col_index = col_index
        term_document_matrix[matrix_row_index][matrix_col_index] = doc_word_counts[key]
    return term_document_matrix

def create_term_context_matrix(line_tuples, vocab, context_window_size=1):
    """Returns a numpy array containing the term context matrix for the input lines.

    Inputs:
      line_tuples: A list of tuples, containing the name of the document and
      a tokenized line from that document.
      vocab: A list of the tokens in the vocabulary

    # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:39 PM.

    Let n = len(vocab).

    Returns:
      tc_matrix: A nxn numpy array where A_ij contains the frequency with which
          word j was found within context_window_size to the left or right of
          word i in any sentence in the tuples.
    """
    m = len(vocab)
    doc_tuples = get_doc_tuples(line_tuples)
    term_context_matrix = np.zeros(shape=(m, m))
    doc_tokens = []
    for key in doc_tuples.keys():
      doc_tokens.append(get_tokens_from_doc_tuples(doc_tuples[key]))
    for tokens in doc_tokens: 
      for i in range(context_window_size, len(tokens)-context_window_size):
        word = tokens[i]
        word_index = vocab.index(word)
        prewindow = tokens[i-context_window_size : i]
        postwindow = tokens[i+1 : i+1+context_window_size]
        context = prewindow + postwindow
        for c in context:
          c_index = vocab.index(c)
          term_context_matrix[word_index][c_index] += 1
    return term_context_matrix

def create_tf_idf_matrix(term_document_matrix):
  """Given the term document matrix, output a tf-idf weighted version.

  See section 6.5 in the textbook.

  Hint: Use numpy matrix and vector operations to speed up implementation.

  Input:
    term_document_matrix: Numpy array where each column represents a document
    and each row, the frequency of a word in that document.

  Returns:
    A numpy array with the same dimension as term_document_matrix, where
    A_ij is weighted by the inverse document frequency of document h.
  """
  m = len(term_document_matrix[:])
  n = len(term_document_matrix[0][:])
  tf_idf_matrix = np.empty([m, n])
  idfs = np.empty([m, 1])
  for i in range(m):
    df = 0
    for j in range(n):
      if term_document_matrix[i][j] > 0:
        df += 1
    idfs[i] = math.log10(m/df)
  tf_idf_matrix = term_document_matrix * idfs
  return tf_idf_matrix

def create_ppmi_matrix(term_context_matrix):
  """Given the term context matrix, output a PPMI weighted version.

  See section 6.6 in the textbook.

  Hint: Use numpy matrix and vector operations to speed up implementation.

  Input:
    term_context_matrix: Numpy array where each column represents a context word
    and each row, the frequency of a word that occurs with that context word.

  Returns:
    A numpy array with the same dimension as term_context_matrix, where
    A_ij is weighted by PPMI.
  """
  w_counts = np.sum(term_context_matrix, axis=1)
  c_counts = np.sum(term_context_matrix, axis=0)
  total_counts = np.sum(w_counts) + np.sum(c_counts)
  m = len(term_context_matrix[:])
  ppmi_matrix = np.empty([m, m])
  for i in range(m):
    p_w = w_counts[i] / total_counts
    for j in range(m):
      if term_context_matrix[i][j] == 0:
        pmi = 0
      else:
        p_w_c = term_context_matrix[i][j] / total_counts
        p_c = c_counts[j] / total_counts
        p = p_w_c / (p_w * p_c)
        pmi = math.log2(p)
      ppmi_matrix[i][j] = max([0, pmi])
  return ppmi_matrix

def compute_cosine_similarity(vector1, vector2):
    """Computes the cosine similarity of the two input vectors.

    Inputs:
      vector1: A nx1 numpy array
      vector2: A nx1 numpy array

    Returns:
      A scalar similarity value.
    """
    # Check for 0 vectors
    if not np.any(vector1) or not np.any(vector2):
        sim = 0

    else:
        sim = 1 - scipy.spatial.distance.cosine(vector1, vector2)

    return sim

def rank_words(target_word_index, matrix):
  """Ranks the similarity of all of the words to the target word using compute_cosine_similarity.

  Inputs:
    target_word_index: The index of the word we want to compare all others against.
    matrix: Numpy matrix where the ith row represents a vector embedding of the ith word.

  Returns:
    A length-n list of integer word indices, ordered by decreasing similarity to the
    target word indexed by word_index
    A length-n list of similarity scores, ordered by decreasing similarity to the
    target word indexed by word_index
  """
  m = len(matrix[:])
  similarity_scores = []
  target_vec = matrix[target_word_index]
  for index, vec in zip(range(m), matrix):
    similarity_value = compute_cosine_similarity(vec, target_vec)
    similarity_scores.append((index, similarity_value))
  similarity_scores.sort(key=lambda x: x[1], reverse=True)
  similarity_word_indices = []
  similarity_word_scores = []
  for tuple in similarity_scores:
    similarity_word_indices.append(tuple[0])
    similarity_word_scores.append(tuple[1])
  return similarity_word_indices, similarity_word_scores

def build_np_array_from_csv_file(file_name, n, m):
  matrix = np.empty([n, m])
  with open(file_name) as f:
    reader = csv.reader(f)
    arr = list(reader)
    matrix = np.array(arr)
  print(file_name, ' csv file read')
  return matrix

def part_1(words):
  tuples, document_names, vocab = read_in_shakespeare()
  print("Computing term document matrix...")
  td_matrix = create_term_document_matrix(tuples, document_names, vocab)

  print("Computing tf-idf matrix...")
  tf_idf_matrix = create_tf_idf_matrix(td_matrix)

  print("Computing term context matrix...")
  tc_matrix = create_term_context_matrix(tuples, vocab, context_window_size=2)

  print("Computing PPMI matrix...")
  ppmi_matrix = create_ppmi_matrix(tc_matrix)

  random_idx = random.randint(0, len(document_names) - 1)

  vocab_to_index = dict(zip(vocab, range(0, len(vocab))))

  for word in words:
    print(
        '\nThe 10 most similar words to "%s" using cosine-similarity on term-document frequency matrix are:'
        % (word)
    )
    ranks, scores = rank_words(vocab_to_index[word], td_matrix)
    for idx in range(0,10):
        word_id = ranks[idx]
        print("%d: %s; %s" %(idx+1, vocab[word_id], scores[idx]))

    print(
        '\nThe 10 most similar words to "%s" using cosine-similarity on term-context frequency matrix are:'
        % (word)
    )
    ranks, scores = rank_words(vocab_to_index[word], tc_matrix)
    for idx in range(0,10):
        word_id = ranks[idx]
        print("%d: %s; %s" %(idx+1, vocab[word_id], scores[idx]))

    print(
        '\nThe 10 most similar words to "%s" using cosine-similarity on tf-idf matrix are:'
        % (word)
    )
    ranks, scores = rank_words(vocab_to_index[word], tf_idf_matrix)
    for idx in range(0,10):
        word_id = ranks[idx]
        print("%d: %s; %s" %(idx+1, vocab[word_id], scores[idx]))

    print(
        '\nThe 10 most similar words to "%s" using cosine-similarity on PPMI matrix are:'
        % (word)
    )
    ranks, scores = rank_words(vocab_to_index[word], ppmi_matrix)
    for idx in range(0,10):
        word_id = ranks[idx]
        print("%d: %s; %s" %(idx+1, vocab[word_id], scores[idx]))
   
def part_2(words):
  tuples = read_in_snli()
  vocab = get_unique_tokens(get_tokens_from_doc_tuples(tuples))
  tc_matrix = create_term_context_matrix(tuples, vocab, context_window_size=5)
  ppmi_matrix = create_ppmi_matrix(tc_matrix)
  vocab_to_index = dict(zip(vocab, range(0, len(vocab))))


  for word in words: 
    print(
      '\nThe 10 most similar words to "%s" using cosine-similarity on PPMI matrix are:'
      % (word)
    )
    if word in vocab_to_index:
      ranks, scores = rank_words(vocab_to_index[word], ppmi_matrix)
      for idx in range(0,20):
        word_id = ranks[idx]
        print("%d: %s; %s" %(idx+1, vocab[word_id], scores[idx]))
    else:
      print(
      '\nThe word "%s" was not found.'
      % (word)
      )

if __name__ == "__main__":
  words_part_1 = [ 'juliet', 'king', 'harry']
  words_part_2 = ['caucasian', 'caucasion', 'batwoman', 'batgirl', 'islamic', 'hijabs', 'hijabis', 'villagers', 'ghetto', 'slum', 'tribe', 'tribal', 'panhandling', 'spy', 'spying', 'mr', 'mrs', 'miss', 'ms']
  identity_labels = [
    'woman',
    'women',
    'man',
    'men',
    'girl',
    'girls',
    'boy',
    'boys',
    'she',
    'he',
    'her',
    'him',
    'his',
    'female',
    'male',
    'mother',
    'father',
    'sister',
    'brother',
    'daughter',
    'son',
    'feminine',
    'masculine',
    'androgynous',
    'black',
    'asian',
    'hispanic',
    'white',
    'african',
    'american',
    'latino',
    'latina',
    'caucasian',
    'africans',
    'australian',
    'australians',
    'asians',
    'european',
    'europeans',
    'chinese',
    'indian',
    'indonesian',
    'brazilian',
    'pakistani',
    'russian',
    'nigerian',
    'japanese',
    'mexican',
    'german',
    'egyptian',
    'ethiopian',
    'turkish',
    'thai',
    'french',
    'italian',
    'korean',
    'spanish',
    'dutch',
    'swiss',
    'saudi',
    'belgian',
    'polish',
    'israeli',
    'irish',
    'greek',
    'mongolian',
    'armenian',
  ]
  # part_1(words_part_1)
  # part_2(words_part_2)
  part_2(identity_labels)