# TODO choose a more robust algorithm to predict tags
from settings import PATHS
from nltk import NaiveBayesClassifier
import pickle


with open(PATHS['DATA'] + '/cache.pickle', 'rb') as f:
	data = pickle.load(f)

print(data)
