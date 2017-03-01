import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
import os
import zipfile as zf
import spacy
from settings import PUNC
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


def timeit(func):
    import time
    def wrapper(*args, **kwargs):
        time_begin = time.time()
        func_output = func(*args, **kwargs)
        time_finish = time.time()
        print('{0} function took {1:0.3f} ms'.format(func.__name__, (time_finish-time_begin)*1000.0), flush=True)
        return func_output
    return wrapper


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
        # html_string = ' '.join(
        #     [string.get_text(strip=True) for string in BeautifulSoup(input_data, 'html5lib').find_all(['p', 'li'])]
        #     )
        html_string = BeautifulSoup(input_data, 'html5lib').get_text(strip=True)
        ascii_string = html_string.encode('utf-8').decode('ascii', 'ignore')
        string = ascii_string.lower().translate(str.maketrans(PUNC, ' '*len(PUNC)))
        return ' '.join(string.split())

    @timeit
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




