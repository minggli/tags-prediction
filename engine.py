from settings import PATHS
from helpers import word_feat
from nltk import NaiveBayesClassifier
from nltk.classify import apply_features
import pickle
import numpy as np
from gensim import models, corpora


with open(PATHS['DATA'] + '/complete_cache.pickle', 'rb') as f:
	data = pickle.load(f)

with open(PATHS['DATA'] + '/test_cache.pickle', 'rb') as f:
	test = pickle.load(f)

# TODO trying latent Dirichlet Allocation (LDA)

# first attempt Navie Bayes

def batch_iterator(np_data, batch_size=10000):

	assert isinstance(np_data, np.ndarray), 'require numpy array.'
	n = len(np_data)
	num_batches = -(-n // batch_size)

	for each_batch in range(num_batches):
		start_index = each_batch * batch_size
		stop_index = min(start_index + batch_size, n)
		if start_index == stop_index:
			break
		else:
			yield num_batches, each_batch, np_data[start_index: stop_index]

def nb_data():

	print('preparing Navie Bayes training data...')

	# iterator = batch_iterator(np_data=data, batch_size=10000)

	# for n, k, data_slice in iterator:
	# 	increment = np.random.permutation([i for i in map(lambda x: tuple((word_feat(x[0].split(), numeric=True), x[1])), data_slice)])
	# 	if k > 0:
	# 		nb_train = np.concatenate((nb_train, increment), axis=0)
	# 	else:
	# 		nb_train = increment

	# 	print('completed preparing {0} of {1}...'.format(k + 1, n), end='\n')

	nb_train = apply_features(lambda x: word_feat(x[0].split(), numeric=True), data, labeled=True)

	# nb_train = np.random.permutation([i for i in map(lambda x: tuple((word_feat(x[0].split(), numeric=True), x[1])), data)])
	# with open(PATHS['DATA'] + '/nb_cache.pickle', 'wb') as f:
	# 	pickle.dump(nb_train, f)
	print('done...total of {0} prepared...'.format(len(nb_train)))
	return nb_train


def train():
	
	nb_train = nb_data()

	print('training Naive Bayes classifer...', flush=True, end='\n')
	clf = NaiveBayesClassifier.train(nb_train)
	print('done')

	return clf

nb_test = np.array([i for i in map(lambda x: word_feat(x[0].split(), numeric=True), test)])


def classify(clf, word_features, decision_boundary=.5, limit=5):
	assert isinstance(clf, NaiveBayesClassifier), 'require nltk classifier'

	obj = clf.prob_classify(word_features)
	options = list()
	keys = list(obj.samples())

	for key in keys:
		prob = obj.prob(key)
		options.append(tuple((key, prob)))

	# sorting pairs in probability descending order
	options.sort(key=lambda x: x[1], reverse=True)
	# only keeping most likely tags
	options = options[:limit]

	return [pair[0] for pair in filter(lambda x: x[1] > decision_boundary, options)]