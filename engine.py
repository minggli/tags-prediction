from settings import PATHS, LIMIT
from helpers import word_feat
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import numpy as np
import pickle
import os
import sys
sys.setrecursionlimit(30000)


with open(PATHS['DATA'] + '/complete_cache.pickle', 'rb') as f:
	data = pickle.load(f)

with open(PATHS['DATA'] + '/test_cache.pickle', 'rb') as f:
	test_data = pickle.load(f)


class TF_IDF(object):
	"""
	using term frequency and inverse document frequency to extract features
	with highest weights
	"""

	def __init__(self, vectorizer, transformer, limit):

		self.__vectorizer__ = vectorizer
		self.__transformer__ = transformer
		self.__limit__ = limit

		self._iterator = None
		self._vectorizer = None
		self._transformer = None
		self._limit = None
		self._feat_names = None
		self._tf_idf_matrix = None

	@property
	def __vectorizer__(self):
		return self._vectorizer

	@__vectorizer__.setter
	def __vectorizer__(self, object):
		if not isinstance(object, sklearn.feature_extraction.text.TfidfVectorizer):
			raise TypeError('requires scikit-learn TfidfVectorizer.')
		else:
			self._vectorizer = object

	@property
	def __transformer__(self):
		return self._transformer

	@__transformer__.setter
	def __transformer__(self, object):
		if not isinstance(object, sklearn.feature_extraction.text.TfidfTransformer):
			raise TypeError('requires scikit-learn TfidfTransformer.')
		else:
			self._transformer = object

	@property
	def __limit__(self):
		return self._limit

	@__limit__.setter
	def __limit__(self, value):
		if not isinstance(value, (int, float)):
			raise TypeError('limit must be a numeric value.')
		elif not int(value) == value:
			raise ValueError('limit must be an integer.')
		elif not 0 < value <= 20:
			raise ValueError('limit must be between 1 and 20.')
		else:
			self._limit = value

	def fit_transform(self, training_set):

		assert isinstance(training_set, np.array)
		assert isinstance(training_set[0], str)

		self._vectorizer.fit(training_set)
		self._feat_names = self._vectorizer.get_feature_names()

		tf_idf_matrix = self._vectorizer.transform(training_set)
		self._tf_idf_matrix = self._transformer.fit_transform(tf_idf_matrix)

		return self

	def __iter__(self):
		"""an iterator to spell out terms with highest weightings"""
		if not self._tf_idf_matrix:
			raise RunTimeError('train TF-IDF algorithm first.')

		densed_documents = self._tf_idf_matrix.todense()
		n = len(densed_documents)

		for doc_id in range(n):

			densed_document = densed_documents[doc_id].tolist()[0]
			phrase_scores = [pair for pair in zip(range(0, len(densed_document)), densed_document) if pair[1] > 0]
			sorted_phrase_scores = sorted(phrase_scores, key=lambda x: x[1], reverse=True)
			named_scores = [self._feat_names[pair[0]] for pair in sorted_phrase_scores][:self._limit]
			feat_label_pair = tuple((word_feat(named_scores, numeric=False), labels[doc_id].split(' ')))

			new_train_set.append(feat_label_pair)




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

	iterator = batch_iterator(np_data=np.array(data), batch_size=10000)
	func = lambda x: x[0]
	func_y = lambda x: x[1]

	for n, k, data_slice in iterator:
		increment = np.array([i for i in map(func, data_slice)])
		increment_y = np.array([i for i in map(func_y, data_slice)])
		if k > 0:
			train = np.concatenate((train, increment), axis=0)
			train_y = np.concatenate((train_y, increment_y), axis=0)
		else:
			train = increment
			train_y = increment_y
		print('completed preparing {0} of {1}...'.format(k + 1, n), end='\n', flush=True)
	print('done...total of {0} prepared...'.format(len(train)), flush=True)
	return train, train_y


def nb_test_data():

	iterator = batch_iterator(np_data=np.array(test_data), batch_size=10000)
	func = lambda x: x[1]

	for n, k, data_slice in iterator:
		increment = np.array([i for i in map(func, data_slice)])
		if k > 0:
			test = np.concatenate((test, increment), axis=0)
		else:
			test = increment
		print('completed preparing {0} of {1}...'.format(k + 1, n), end='\n', flush=True)
	print('done...total of {0} prepared...'.format(len(test)), flush=True)
	return test

# if os.path.exists(PATHS['DATA'] + '/tf_idf_matrix.pickle'):

corpus, labels = nb_data()
# nb_test = nb_test_data()
# corpus = np.concatenate((corpus, nb_test), axis=0)

tf = TfidfVectorizer(
	input='content',
	encoding='utf-8',
	ngram_range=(1, 1),
	strip_accents='ascii',
	analyzer='word',
	stop_words='english')


tf.fit(corpus)
feature_names = tf.get_feature_names()
tf_idf_matrix = tf.transform(corpus)
tf_idf_matrix_transformed = TfidfTransformer().fit_transform(tf_idf_matrix)

# 	with open(PATHS['DATA'] + '/tf_idf_matrix.pickle', 'wb') as f:
# 		pickle.dump(tf_idf_matrix_transformed, f)

# 	with open(PATHS['DATA'] + '/word_feats.pickle', 'wb') as f:
# 		pickle.dump(feature_names, f)

# else:

# 	with open(PATHS['DATA'] + '/tf_idf_matrix.pickle', 'rb') as f:
# 		tf_idf_matrix_transformed = pickle.load(f)

# 	with open(PATHS['DATA'] + '/word_feats.pickle', 'rb') as f:
# 		feature_names = pickle.load(f)

new_train_set = list()

densed_documents = tf_idf_matrix_transformed.todense()
for doc in range(len(densed_documents)):

	densed_document = densed_documents[doc].tolist()[0]
	phrase_scores = [pair for pair in zip(range(0, len(densed_document)), densed_document) if pair[1] > 0]
	sorted_phrase_scores = sorted(phrase_scores, key=lambda x: x[1], reverse=True)
	named_scores = [feature_names[pair[0]] for pair in sorted_phrase_scores][:LIMIT]
	feat_label_pair = tuple((word_feat(named_scores, numeric=False), labels[doc].split(' ')))

	new_train_set.append(feat_label_pair)


print(new_train_set)
# for word, score in named_scores:
# 	print('{0: <20} {1}'.format(word, score))
