from settings import PATHS, LIMIT
from tf_idf import TF_IDF
from helpers import sort_coo
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
import numpy as np
import pickle
import os
import sys

with open(PATHS['DATA'] + '/train_data.npz', 'rb') as f:
    npz_object = np.load(f)
    train_features = npz_object['train_features']
    train_labels = npz_object['train_labels']

tf_vector = TfidfVectorizer(
	input='content',
	encoding='utf-8',
	ngram_range=(1, 1),
	strip_accents='ascii',
	analyzer='word',
	stop_words='english'
	)

tf_trans = TfidfTransformer()

train_tfidf = TF_IDF(vectorizer=tf_vector, transformer=tf_trans, limit=LIMIT)
train_tfidf.fit_transform(train_features)
del train_features

binarizer = MultiLabelBinarizer(sparse_output=False).fit(train_labels)
onehot_encoded_labels = binarizer.transform(train_labels)

# TODO investigate how to manipulate sparse matrix so as to feed sparse matrix from tf-idf module into vectorizer directly.
vectorized_train = list(train_tfidf)
# use trained vectorizer to transform test set

cv = CountVectorizer()
cv.fit(vectorized_train)

estimator = RandomForestClassifier(
	n_estimators=10,
	criterion='gini',
	max_depth=None,
	min_samples_split=2,
	min_samples_leaf=1,
	min_weight_fraction_leaf=0.0,
	max_features='auto',
	max_leaf_nodes=None,
	min_impurity_split=1e-07,
	bootstrap=True,
	oob_score=False,
	random_state=None,
	verbose=0,
	warm_start=False,
	class_weight=None
	)

vectorized_train = cv.transform(vectorized_train)
print('beginning training Multinomial Classifier with multi-label strategy...', end='', flush=True)
OvR = OneVsRestClassifier(estimator, n_jobs=-1)
OvR.fit(vectorized_train, onehot_encoded_labels)
print('done', flush=True)

del train_labels
del vectorized_train

with open(PATHS['DATA'] + '/test_data.npz', 'rb') as f:
    npz_object = np.load(f)
    test_features = npz_object['test_features']

test_tfidf = TF_IDF(vectorizer=tf_vector, transformer=tf_trans, limit=LIMIT)
test_tfidf.fit_transform(test_features)

del test_features

vectorized_test = list(test_tfidf)
# Named features not encountered during fit or fit_transform will be silently ignored.
vectorized_test = cv.transform(vectorized_test)

print('beginning predicting multiple labels on test set.', flush=True)
predicted_onehot_matrix = OvR.predict(vectorized_test)
# TODO add probability threshold to OvR to prevent unlikely tags
del OvR

predicted_labels = binarizer.inverse_transform(predicted_onehot_matrix)
