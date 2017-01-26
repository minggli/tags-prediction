from settings import PATHS
from nltk import NaiveBayesClassifier
import pickle
from gensim import models, corpora
from helpers import word_feat
import numpy as np

with open(PATHS['DATA'] + '/complete_cache.pickle', 'rb') as f:
	data = pickle.load(f)

with open(PATHS['DATA'] + '/test_cache.pickle', 'rb') as f:
	test = pickle.load(f)
# TODO trying latent Dirichlet Allocation (LDA)


# first attempt Navie Bayes

n = int(2e2)
nb_train = np.random.permutation([i for i in map(lambda x: tuple((word_feat(x[0].split(), numeric=True), x[1])), data[:n])])
print('training Naive Bayes classifer...', flush=False, end='')
clf = NaiveBayesClassifier.train(nb_train)
print('done')

def classify(clf, string, decision_boundary=.5):

	assert isinstance(clf, NaiveBayesClassifier), 'require nltk classifier'

	obj = clf.prob_classify(string)
	options = list()
	keys = list(obj.samples())

	for key in keys:
		prob = obj.prob(key)
		options.append(tuple(key, prob))

	# sorting pairs in probability descending order
	options.sort(key=lambda x: x[1], reverse=True)

	return [i for i in filter(lambda x: x[1] > decision_boundary, options)]

nb_test = np.random.permutation([i for i in map(lambda x: (word_feat(x[0].split(), numeric=True)), test)])


print(test)