import argparse
from collections import Counter
from n_grams import NGrams
import os

data_file_paths = {
    'en': 'datasets/hw3_data/training.en',
    'es': 'datasets/hw3_data/training.es',
    'de': 'datasets/hw3_data/training.de',
    'test': 'datasets/hw3_data/test'
}

parser = argparse.ArgumentParser()
parser.add_argument('--smoothed', default=None, action='store_true')
parser.add_argument('--interpolation', default=None, action='store_true')                    
args = parser.parse_args() 
model_type = 'unsmoothed'
if(args.smoothed != None):
    model_type = 'smoothed'
if(args.interpolation != None):
    model_type = 'interpolation'

data_file_path_keys = list(data_file_paths.keys())
n_gram_models = {}
for key in data_file_path_keys:
    if key == 'test':
        data_file_path = data_file_paths[key]
        print('current file path: ', data_file_path)
        f = open(data_file_path)
        unigrams = NGrams(1, f)
        bigrams = NGrams(2, f)
        trigrams = NGrams(3, f) 
        print('uni**', unigrams.count, len(unigrams.values), unigrams.values)
        print('bi**', bigrams.count, len(bigrams.values), bigrams.values)
        print('tri**', trigrams.count, len(trigrams.values), trigrams.values)

