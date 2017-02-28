import pandas as pd
import os
import zipfile as zf
import spacy
from settings import PUNC
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


def unzip_folder(path, exclude=None):
    """read zip files and load uncompressed csv files into a list of panda dataframes"""

    zip_files = [f for f in os.listdir(path) if f.endswith('.zip')]
    list_dataframes = list()

    for i in zip_files:
        zip_obj = zf.ZipFile(file=path + i)
        filename = zip_obj.namelist()[0]
        if exclude and filename in exclude:
            pass
        else:
            print('unzipping {}...'.format(filename))
            df = pd.read_csv(filepath_or_buffer=zip_obj.open(filename), index_col='id')
            zip_obj.close()
            list_dataframes.append(df)

    return list_dataframes


def test(test_string='James is travelling to London this Sunday. We are too.'):

    doc = spacy.en.English(test_string)

    for sentence in doc.sents:
        print(sentence)

    for token in doc:
        print(token, token.tag, token.tag_, token.lemma, token.lemma_, token.pos, token.pos_)

    for ent in doc.ents:
        print(ent, ent.label, ent.label_)

    for np in doc.noun_chunks:
        print(np)

    print(doc[0].similarity(doc[6]))


def word_feat(words, numeric=True):
    return AdditiveDict([(word, True) for word in words]) if numeric else dict([(word, True) for word in words])


class Preprocessor(object):

    status = {
        True: 'Processed',
        False: 'Unprocessed'
    }

    def __init__(self, df):
        assert isinstance(df, pd.DataFrame), 'input require Pandas DataFrame object.'
        self._df = df
        self.data = None
        self.is_processed = False

    def _parse(self, input_data):
        """extract texts from html and punctuations"""
        html_string = ' '.join(
            [string.get_text(strip=True) for string in BeautifulSoup(input_data, 'html5lib').find_all(['p', 'li'])]
            )
        ascii_string = html_string.encode('utf-8').decode('ascii', 'ignore')
        string = ascii_string.lower().translate(str.maketrans(PUNC, ' '*len(PUNC)))
        return ' '.join(string.split())

    def process(self):
        print('\npre-precessing texts...', flush=True)
        source = self._df.copy()
        source.ix[:, ~source.columns.isin(['tags'])] = \
            source.ix[:, ~source.columns.isin(['tags'])].applymap(lambda x: self._parse(x))
        self.data = source
        self.is_processed = True

    def __iter__(self):
        if not self.is_processed:
            self.process()
        for row in self.data.itertuples():
            yield row

    def __str__(self):
        return '{} {} object'.format(self.status[self.is_processed], self.__class__.__name__)

    def __len__(self):
        return self._df.shape[0]

    def __gt__(self, other):
        return self.__len__() > other.__len__()

    def __ge__(self, other):
        return self.__len__() >= other.__len__()


class AdditiveDict(dict):

    def __init__(self, iterable=None):
        if not iterable:
            pass
        else:
            assert hasattr(iterable, '__iter__')
            for i in iterable:
                self.__setitem__(i[0], 0)

    def __missing__(self, key):
        return 0

    def __setitem__(self, key, value):
        super(AdditiveDict, self).__setitem__(key, self.__getitem__(key) + 1)


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
            
            named_scores = [
            self._feat_names[pair[0]] for pair in 
            sorted(phrase_scores, key=lambda x: x[1], reverse=True)
            ][:self._limit]

            yield word_feat(named_scores, numeric=False)


