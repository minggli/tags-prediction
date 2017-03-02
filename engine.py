from settings import PATHS, LIMIT
from tf_idf import TF_IDF
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
import numpy as np
import pickle
import os
import sys

with open(PATHS['DATA'] + '/train_data.npz', 'rb') as f:
    npz_object = np.load(f)
    train_features = npz_object['train_features']
    train_labels = npz_object['train_labels']

with open(PATHS['DATA'] + '/test_data.npy', 'rb') as f:
    npz_object = np.load(f)
    test_features = npz_object['test_features']

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
train_word_feats = train_tfidf.fit_transform(train_features)
del train_features
binarizer = MultiLabelBinarizer(sparse_output=False).fit(train_labels)
labels_classes = binarizer.classes_
onehot_encoded_labels = binarizer.transform(train_labels)

# for i in train_tfidf._tf_idf_matrix:
# 	print(i)
# 	break
# sys.exit()

# TODO investigate how to manipulate sparse matrix so as to feed sparse matrix from tf-idf module into vectorizer directly.
vectorized_train = list(train_word_feats)
# use trained vectorizer to transform test set

cv = CountVectorizer()
cv.fit(vectorized_train)

vectorized_train = cv.transform(vectorized_train)
print('beginning training Multinomial Naive Bayes with multi-label strategy.', end='', flush=True)
OvR = OneVsRestClassifier(MultinomialNB(), n_jobs=-1)
OvR.fit(vectorized_train, onehot_encoded_labels)
print('done', flush=True)

del train_labels
del vectorized_train

with open(PATHS['DATA'] + '/test_features.npy', 'rb') as f:
    test_features = np.load(f)

test_tfidf = TF_IDF(vectorizer=tf_vector, transformer=tf_trans, limit=LIMIT)
test_word_feats = test_tfidf.fit_transform(test_features)

del test_features

vectorized_test = list(test_word_feats)
# Named features not encountered during fit or fit_transform will be silently ignored.
vectorized_test = cv.transform(vectorized_test)

print('beginning predicting multiple labels on test set.', flush=True)
predicted_onehot_matrix = OvR.predict(vectorized_test)
# TODO add probability threshold to OvR to prevent unlikely tags
del OvR

predicted_labels = binarizer.inverse_transform(predicted_onehot_matrix)
