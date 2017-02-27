from settings import PATHS
from helpers import word_feat
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
import pickle
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

		print('completed preparing {0} of {1}...'.format(k + 1, n), end='\n')

	print('done...total of {0} prepared...'.format(len(train)))

	return train

corpus = nb_data()

corpus[:10]

tf = TfidfVectorizer(
	input='content',
	encoding='utf-8',
	ngram_range=(1, 3),
	strip_accents='ascii',
	analyzer='word',
	stop_words='english'
	)

tf_idf_matrix = tf.fit_transform(corpus)

feature_names = tf.get_feature_names()

print(np.random.permutations(feature_names)[:10])

