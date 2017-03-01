from helpers import Preprocessor, unzip_folder, test, timeit
from settings import PATHS, PUNC, TextMining, TrainFiles
import spacy
import pandas as pd
import pickle
import sys

__author__ = 'Ming Li'

TRAIN = True if 'TRAIN' in map(str.upper, sys.argv[1:]) else False
TEST = True if 'TEST' in map(str.upper, sys.argv[1:]) else False
YIELD = True if 'YIELD' in map(str.upper, sys.argv[1:]) else False

if __name__ == '__main__':

    if TRAIN:
        nlp = spacy.load('en')
        list_of_dataframes = unzip_folder(PATHS['DATA'], exclude=['sample_submission.csv', 'test.csv'])
        df = pd.concat(objs=list_of_dataframes, ignore_index=True)
        train_texts = Preprocessor(df)
        data = list(train_texts)

        with open(PATHS['DATA'] + '/complete_cache.pickle', 'wb') as f:
            pickle.dump(data, f)

    if TEST:
        nlp = spacy.load('en')
        test = unzip_folder(PATHS['DATA'], exclude=TrainFiles + ['sample_submission.csv'])[0]
        test_texts = Preprocessor(test)
        data = list(test_texts)

        with open(PATHS['DATA'] + '/test_cache.pickle', 'wb') as f:
            pickle.dump(data, f)

    if YIELD:
        from output import generate_submission
        generate_submission()
