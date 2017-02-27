from settings import PATHS
from helpers import word_feat
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
import pickle
import os
import sys
sys.setrecursionlimit(30000)
import numpy as np


with open(PATHS['DATA'] + '/complete_cache.pickle', 'rb') as f:
	data = pickle.load(f)

with open(PATHS['DATA'] + '/test_cache.pickle', 'rb') as f:
	test = pickle.load(f)


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

	for n, k, data_slice in iterator:
		increment = np.random.permutation([i for i in map(func, data_slice)])
		if k > 0:
			train = np.concatenate((train, increment), axis=0)
		else:
			train = increment

		print('completed preparing {0} of {1}...'.format(k + 1, n), end='\n', flush=True)

	print('done...total of {0} prepared...'.format(len(train)), flush=True)

	return train

# corpus = nb_data()

# tf = TfidfVectorizer(
# 	input='content',
# 	encoding='utf-8',
# 	ngram_range=(1, 1),
# 	strip_accents='ascii',
# 	analyzer='word',
# 	stop_words='english'
# 	)

# tf.fit(corpus)

# feature_names = tf.get_feature_names()

if not os.path.exists(PATHS['DATA'] + '/tf_idf_matrix.pickle'):

	tf_idf_matrix = tf.transform(corpus)

	tf_idf_matrix_transformed = TfidfTransformer().fit_transform(tf_idf_matrix)

	with open(PATHS['DATA'] + '/tf_idf_matrix.pickle', 'wb') as f:
		pickle.dump(tf_idf_matrix_transformed, f)

else:

	with open(PATHS['DATA'] + '/tf_idf_matrix.pickle', 'rb') as f:
		tf_idf_matrix_transformed = pickle.load(f)

densed_document = tf_idf_matrix_transformed.todense()[0].tolist()[0]
phrase_scores = [pair for pair in zip(range(0, len(densed_document)), densed_document) if pair[1] > 0]
sorted_phrase_scores = sorted(phrase_scores, key=lambda x: x[1], reverse=True)
print(sorted_phrase_scores)

