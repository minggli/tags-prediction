# from engine import clf
from helpers import Preprocessor, unzip_folder, test
from settings import PATHS, PUNC, TextMining
import spacy
import pickle


train_files = ['biology.csv', 'cooking.csv', 'crypto.csv', 'diy.csv', 'robotics.csv', 'travel.csv']

test = unzip_folder(PATHS['DATA'], exclude=train_files + ['sample_submission.csv'])[0]
print(test)

texts = Preprocessor(test)

clf.prob_classify()