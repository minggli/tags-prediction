# TODO choose a more robust algorithm to predict tags

from nltk import NaiveBayesClassifier
from .main import data

clf = NaiveBayesClassifier()
