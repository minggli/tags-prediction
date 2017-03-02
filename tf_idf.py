"""
    using term frequency and inverse document frequency to extract features
    with highest weights
"""

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from helpers import word_feat, timeit
import numpy as np


class TF_IDF(object):

    def __init__(self, vectorizer, transformer, limit):

        self._iterator = None
        self._vectorizer = None
        self._transformer = None
        self._limit = None
        self._feat_names = None
        self._tf_idf_matrix = None

        self.vectorizer = vectorizer
        self.transformer = transformer
        self.limit = limit

    @property
    def vectorizer(self):
        return self._vectorizer

    @vectorizer.setter
    def vectorizer(self, obj):
        if not isinstance(obj, TfidfVectorizer):
            raise TypeError('requires scikit-learn TfidfVectorizer.')
        else:
            self._vectorizer = obj

    @property
    def transformer(self):
        return self._transformer

    @transformer.setter
    def transformer(self, obj):
        if not isinstance(obj, TfidfTransformer):
            raise TypeError('requires scikit-learn TfidfTransformer.')
        else:
            self._transformer = obj

    @property
    def limit(self):
        return self._limit

    @limit.setter
    def limit(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError('limit must be a numeric value.')
        elif not int(value) == value:
            raise ValueError('limit must be an integer.')
        elif not 0 < value <= 20:
            raise ValueError('limit must be between 1 and 20.')
        else:
            self._limit = value

    @timeit
    def fit_transform(self, data_set):

        assert isinstance(data_set, np.ndarray)
        assert isinstance(data_set[0], str)

        self._vectorizer.fit(data_set)
        self._feat_names = self._vectorizer.get_feature_names()

        tf_idf_matrix = self._vectorizer.transform(data_set)
        self._tf_idf_matrix = self._transformer.fit_transform(tf_idf_matrix)

        return self

    def __iter__(self):
        """an iterator to spell out terms with highest weightings"""
        if self._tf_idf_matrix is None:
            raise RunTimeError('train TF-IDF algorithm first.')
        elif self._tf_idf_matrix is not None:
            n = self._tf_idf_matrix.shape[0]
            for doc_id in range(n):
                densed_document = self._tf_idf_matrix[doc_id].todense().tolist()[0]
                # densed_document = densed_documents[doc_id].tolist()[0]
                phrase_scores = [pair for pair in zip(range(0, len(densed_document)), densed_document) if pair[1] > 0]
                named_scores = set([
                self._feat_names[pair[0]] for pair in 
                sorted(phrase_scores, key=lambda x: x[1], reverse=True)
                ][:self._limit])
                output_string = ' '.join(named_scores)
                print('post-processing {} of {} documents with TF-IDF scores.'.format(doc_id + 1, n), flush=True)
                yield output_string