from engine import clf, classify, nb_test
from helpers import Preprocessor, unzip_folder, test
from settings import PATHS, PUNC, TextMining
import spacy
import pickle



test = unzip_folder(PATHS['DATA'], exclude=train_files + ['sample_submission.csv'])[0]
print(test)

texts = Preprocessor(test)

clf.prob_classify()