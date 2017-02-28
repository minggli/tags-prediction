from settings import PATHS, LIMIT
from helpers import word_feat
from tf_idf import TF_IDF
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import numpy as np
import pickle
import pandas as pd
import os
import sys
sys.setrecursionlimit(30000)


with open(PATHS['DATA'] + '/complete_cache.pickle', 'rb') as f:
	train_data = pickle.load(f)

with open(PATHS['DATA'] + '/test_cache.pickle', 'rb') as f:
	test_data = pickle.load(f)


def batch_iterator(data, test_set=False, batch_size=10000):

	data = np.array(data)
	n = len(data)
	num_batches = -(-n // batch_size)

	func_feat = lambda x: x[0]
	func_label = lambda x: x[1].split()

	for each_batch in range(num_batches):
		start_index = each_batch * batch_size
		stop_index = min(start_index + batch_size, n)
		if start_index >= stop_index:
			break
		else:
			if not test_set:
				# TODO check if map iter can be transformed by array constructor or not
				feature_slice = np.array(
					list(map(func_feat, data[start_index: stop_index]))
					)
				label_slice = np.array(
					list(map(func_label, data[start_index: stop_index]))
					)
				print('preparing {0} of {1}...'.format(each_batch + 1, num_batches), end='\n', flush=True)
				yield feature_slice, label_slice
			elif test_set:
				feature_slice = np.array(
					list(map(lambda x: x[1], data[start_index: stop_index]))
					)
				print('preparing {0} of {1}...'.format(each_batch + 1, num_batches), end='\n', flush=True)
				yield feature_slice


test_features = None
for feat in batch_iterator(data=test_data, test_set=True, batch_size=10000):
	if test_features is None:
		test_features = feat
		break
	else:
		test_features = np.concatenate((test_features, feat), axis=0)

train_features = None
train_labels = None
for feat, label in batch_iterator(data=train_data, test_set=False, batch_size=10000):
	if train_features is None or train_labels is None:
		train_features, train_labels = feat, label
		break
	else:
		# TODO better way to address performance issue
		train_features = np.concatenate((train_features, feat), axis=0)
		train_labels = np.concatenate((train_labels, label), axis=0)


tf_vector = TfidfVectorizer(
	input='content',
	encoding='utf-8',
	ngram_range=(1, 1),
	strip_accents='ascii',
	analyzer='word',
	stop_words='english'
	)

tf_trans = TfidfTransformer()

tf = TF_IDF(vectorizer=tf_vector, transformer=tf_trans, limit=LIMIT)

# tf.fit_transform(test_features)
tf.fit_transform(train_features)

labels = [phase for multilabel in train_labels for phase in multilabel]

a = list(tf)
print(len(a))

dv = DictVectorizer(sparse=False)
dv.fit(a)
a = dv.transform(a)

# labels = pd.Series(data=train_labels, name='tags')
binarizer = MultiLabelBinarizer(sparse_output=False).fit(train_labels)
onehot_labels = binarizer.transform(train_labels)
classes = binarizer.classes_
# print(binarizer.classes_[585], binarizer.classes_[49], binarizer.classes_[610], binarizer.classes_[518])

OvR = OneVsRestClassifier(MultinomialNB())
OvR.fit(a, onehot_labels)


