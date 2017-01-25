from settings import PATHS
from nltk import NaiveBayesClassifier
import pickle
from gensim import models, corpora

with open(PATHS['DATA'] + '/cache.pickle', 'rb') as f:
	data = pickle.load(f)

# trying Latent Latent Dirichlet Allocation (LDA)

def word_feat(words):
	return {word: True for word in words}

nb_train = [i for i in map(lambda x: tuple((word_feat(x[0].split()), x[1])), data[:20])]

print(nb_train[0])

clf = NaiveBayesClassifier.train(nb_train)

