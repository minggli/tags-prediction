import pandas as pd
import os
import zipfile as zf
import spacy
from settings import PUNC
import numpy as np
import re
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


def unzip_folder(path):
    """read zip files and load uncompressed csv files into a list of panda dataframes"""

    zip_files = [f for f in os.listdir(path) if f.endswith('.zip')]
    list_dataframes = list()

    for i in zip_files:
        zip_obj = zf.ZipFile(file=path + i)
        filename = zip_obj.namelist()[0]
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


class CleansedDataIter(object):

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
        html_string = BeautifulSoup(input_data, 'html5lib').text
        string = html_string.lower().translate(str.maketrans('', '', PUNC))
        return string

    def process(self):
        print('\npre-precessing texts...', flush=True)
        self.data = self._df.applymap(lambda x: self._parse(x))
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
