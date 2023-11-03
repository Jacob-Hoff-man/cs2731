import argparse
from n_grams import NGrams
import numpy as np
import pandas as pd
import math

data_file_paths = {
    'en': 'datasets/hw3_data/training.en',
    'es': 'datasets/hw3_data/training.es',
    'de': 'datasets/hw3_data/training.de',
    'test': 'datasets/hw3_data/test'
}

def convert_probability_model_to_dataframe(arr, index, columns):
    return pd.DataFrame(arr, index=index, columns=columns)

def save_dataframe_to_file(df, file_name):
    df.to_csv(file_name, index=True, header=True)

def convert_bigram_tuples_to_strings(tuples_list):
    bigrams = []
    for tuple in tuples_list:
        bigram = tuple[0] + tuple[1]
        bigrams.append(bigram)
    return bigrams

def get_unigrams_probabilities(unigrams):
    probability_model = np.zeros((len(unigrams.vocab)))
    for i in range(len(unigrams.vocab)):
        gram = unigrams.vocab[i]
        num = unigrams.counts[gram]
        den = unigrams.size
        probability = num / den
        probability_model[i] = probability

    if (debug != None):
        print('unigrams total probability: ', np.sum(probability_model))

    return probability_model

def get_bigrams_probabilities(bigrams, unigrams):
    vocab_count = len(bigrams.vocab)
    probability_model = np.zeros((vocab_count, vocab_count))
    for i in range(vocab_count):
        word = bigrams.vocab[i]
        for j in range(len(bigrams.vocab)):
            bigram = (bigrams.vocab[j], word)
            num = bigrams.counts[bigram]
            den = unigrams.counts[word]
            probability = num / den
            probability_model[i][j] = probability

    if (debug != None):
        # print('bigrams total probability', np.sum(probability_model, axis=1))
        print('bigrams total probabilities summed / vocab_count: ', np.sum(probability_model) / vocab_count)

    return probability_model

def get_trigrams_probabilities(trigrams, bigrams):
    vocab_count = len(trigrams.vocab)
    bigram_words = list(bigrams.counts.keys())
    bigram_words_count = len(bigram_words)
    probability_model = np.zeros((bigram_words_count, vocab_count))
    for i in range(vocab_count):
        word = trigrams.vocab[i]
        for j in range(bigram_words_count):
            bigram = bigram_words[j]
            trigram = bigram + (word,)
            num = trigrams.counts[trigram]
            den = bigrams.counts[bigram]
            probability = num / den
            probability_model[j][i] = probability
    
    if (debug != None):
        # print('trigrams total probability', np.sum(probability_model, axis=0))
        print('trigrams total probabilities summed / bigram_words_count: ', np.sum(probability_model) / bigram_words_count)
    
    return probability_model

def get_interpolated_bigrams_probabilities(bigrams, unigrams, w1 = 1 / 2, w2 = 1 / 2):
    vocab_count = len(bigrams.vocab)
    probability_model = np.zeros((vocab_count, vocab_count))
    for i in range(vocab_count):
        word = bigrams.vocab[i]
        for j in range(len(bigrams.vocab)):
            bigram = (bigrams.vocab[j], word)
            bigram_prob = bigrams.counts[bigram] / unigrams.counts[word] * w1
            unigram_prob = unigrams.counts[word] / unigrams.size * w2
            probability = bigram_prob + unigram_prob
            # print('bi_prob', bigram_prob, 'uni_prob', unigram_prob)
            probability_model[i][j] = probability 

    if (debug != None):
        # print('bigrams total probability', np.sum(probability_model, axis=1))
        print('bigrams interpolated total probabilities summed / vocab_count: ', np.sum(probability_model) / vocab_count)

    return probability_model

def get_interpolated_trigrams_probabilities(trigrams, bigrams, unigrams, w1 = 1 / 3, w2 = 1 / 3, w3 = 1 / 3):
    vocab_count = len(trigrams.vocab)
    bigram_words = list(bigrams.counts.keys())
    bigram_words_count = len(bigram_words)
    probability_model = np.zeros((bigram_words_count, vocab_count))
    for i in range(vocab_count):
        word = trigrams.vocab[i]
        for j in range(bigram_words_count):
            bigram = bigram_words[j]
            trigram = bigram + (word,)
            trigram_prob = trigrams.counts[trigram] / bigrams.counts[bigram] * w1
            new_bigram = (bigram[1], word)
            bigram_prob = bigrams.counts[new_bigram] / unigrams.counts[word] * w2
            unigram_prob = unigrams.counts[word] / unigrams.size * w3
            # print('tri_prob', trigram_prob, 'bi_prob', bigram_prob, 'uni_prob', unigram_prob)
            probability = trigram_prob + bigram_prob + unigram_prob
            probability_model[j][i] = probability
    
    if (debug != None):
        # print('trigrams interpolated total probability', np.sum(probability_model, axis=1))
        print('trigrams interpolated total probabilities summed / bigram_words_count: ', np.sum(probability_model) / bigram_words_count)
    
    return probability_model

def perplexity(text, model, n):
    """ Args:
                text: a string of characters
                model: a matrix or df of the probabilities with rows as prefixes, columns as suffixes.
                You can modify this depending on how you set up your model.
                n: n-gram order of the model

            Acknowledgment: 
        https://towardsdatascience.com/perplexity-intuition-and-derivation-105dd481c8f3 
        https://courses.cs.washington.edu/courses/csep517/18au/
        ChatGPT with GPT-3.5
        """

    # FILL IN: Remove any unseen characters from the text that have no unigram probability in the language
    text = text[:-1] #removing new line character
    text = str.lower(text)
    text = '<s> ' + text + ' </s>'
    
    N = len(text)
    char_probs = []
    for i in range(n-1, N):
        prefix = text[i-n+1:i]
        suffix = text[i]
        # FILL IN: look up the probability in the model of the suffix given the prefix
        if n == 1 and suffix in model.index:
            prob = model.loc[suffix].iloc[0]
        elif prefix in model.index and suffix in model.columns:
            prob = model.loc[prefix, suffix]
        if prob != 0:
            char_probs.append(math.log2(prob))
            neg_log_lik = -1 * sum(char_probs) # negative log-likelihood of the text
            ppl = 2 ** (neg_log_lik/(N - n + 1)) # 2 to the power of the negative log likelihood of the words divided by #ngrams
    return ppl

def 
# main
# cli args
parser = argparse.ArgumentParser()
parser.add_argument('--words', default=None, action='store_true')
parser.add_argument('--save', default=None, action='store_true')
parser.add_argument('--debug', default=None, action='store_true')
parser.add_argument('--smoothed', default=None, action='store_true')
parser.add_argument('--interpolation', default=None, action='store_true')                    
args = parser.parse_args() 
model_type = 'unsmoothed'
debug = None
save = None
token_type = 'char'
if args.words != None:
    token_type = 'words'
if args.save != None:
    save = True
if args.debug != None:
    debug = True
if args.smoothed != None:
    model_type = 'smoothed'
if args.interpolation != None:
    model_type = 'interpolation'

# get n_gram_models for files
data_file_path_keys = list(data_file_paths.keys())
n_gram_models = {}
for key in data_file_path_keys:
    data_file_path = data_file_paths[key]
    if (debug != None):
        print('current file path: ', data_file_path)
    f = open(data_file_path)
    unigrams = NGrams(1, f, token_type)
    bigrams = NGrams(2, f, token_type)
    trigrams = NGrams(3, f, token_type)
    n_gram_models[key] = (unigrams, bigrams, trigrams)

# get unigram, bigram, and trigram probabilities for each file
n_gram_probability_model_dfs = {}
for key in data_file_path_keys:
    if key != 'test':
        unigrams = n_gram_models[key][0]
        bigrams =  n_gram_models[key][1]
        trigrams =  n_gram_models[key][2]
        unigrams_probabilities_model = get_unigrams_probabilities(unigrams)
        if model_type == 'interpolation':
            bigrams_probabilities_model = get_interpolated_bigrams_probabilities(bigrams, unigrams)
            trigrams_probabilities_model = get_interpolated_trigrams_probabilities(trigrams, bigrams, unigrams)   
        else:
            bigrams_probabilities_model = get_bigrams_probabilities(bigrams, unigrams)
            trigrams_probabilities_model = get_trigrams_probabilities(trigrams, bigrams)

        unigrams_prob_model_df = convert_probability_model_to_dataframe(
            unigrams_probabilities_model,
            unigrams.vocab,
            None
        )
        bigrams_prob_model_df = convert_probability_model_to_dataframe(
            bigrams_probabilities_model,
            bigrams.vocab,
            bigrams.vocab
        )
        trigrams_prob_model_df = convert_probability_model_to_dataframe(
            trigrams_probabilities_model,
            convert_bigram_tuples_to_strings(list(bigrams.counts.keys())),
            trigrams.vocab
        )
        n_gram_probability_model_dfs[key] = (
            unigrams_prob_model_df,
            bigrams_prob_model_df,
            trigrams_prob_model_df
        )
        if save != None:
            if (model_type == 'interpolation'):
                print('saving', key, 'interpolated probabilities models to file')
                unigram_file_name = 'hw3/' + key + '_unigrams.csv'
                bigram_file_name = 'hw3/' + key + '_interpolated_bigrams.csv'
                trigram_file_name = 'hw3/' + key + '_interpolated_trigrams.csv'
            else:
                print('saving', key, ' probabilities models to file')
                unigram_file_name = 'hw3/' + key + '_unigrams.csv'
                bigram_file_name = 'hw3/' + key + '_bigrams.csv'
                trigram_file_name = 'hw3/' + key + '_trigrams.csv'
            save_dataframe_to_file(unigrams_prob_model_df, unigram_file_name)
            save_dataframe_to_file(bigrams_prob_model_df, bigram_file_name)
            save_dataframe_to_file(trigrams_prob_model_df, trigram_file_name)

# get perplexity for all lines in test file
f = open(data_file_paths['test'])
for key in data_file_path_keys:
    f.seek(0)
    if key != 'test':
        unigrams_model_df = n_gram_probability_model_dfs[key][0]
        bigrams_model_df = n_gram_probability_model_dfs[key][1]
        trigrams_model_df = n_gram_probability_model_dfs[key][2]
        total_unigram_perplexity = 0
        total_bigram_perplexity = 0
        total_trigram_perplexity = 0
        string_count = 0
        for content in f:
            string_count = string_count + 1
            total_unigram_perplexity = perplexity(content, unigrams_model_df, 1)
            total_bigram_perplexity = perplexity(content, bigrams_model_df, 2)
            total_trigram_perplexity = perplexity(content, trigrams_model_df, 3)
        average_unigram_perplexity = total_unigram_perplexity / string_count
        average_bigram_perplexity = total_bigram_perplexity / string_count
        average_trigram_perplexity = total_trigram_perplexity / string_count
        print(key, ' unigram model average perplexity: ', average_unigram_perplexity)
        print(key, ' bigram model average perplexity: ', average_bigram_perplexity)
        print(key, ' trigram model average perplexity: ', average_trigram_perplexity)



