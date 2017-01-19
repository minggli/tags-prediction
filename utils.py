import pandas as pd
import os
import zipfile as zf
import spacy
from settings import PUNC
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


def pipeline(lang='en'):
    """construct a language process pipeline"""
    return spacy.load(lang)


def test(test_string='James is travelling to London this Sunday. We are too.'):

    doc = pipeline()(test_string)

    for sentence in doc.sents:
        print(sentence)

    for token in doc:
        print(token, token.tag, token.tag_, token.lemma, token.lemma_, token.pos, token.pos_)

    for ent in doc.ents:
        print(ent, ent.label, ent.label_)

    for np in doc.noun_chunks:
        print(np)

    print(doc[0].similarity(doc[6]))


class Cleanse(object):

    status = {
        True: 'Processed',
        False: 'Unprocessed'
    }

    def __init__(self, df):
        assert isinstance(df, pd.DataFrame), 'input require Pandas DataFrame object.'
        self._df = df
        self._clean_df = None

    def _parse(self, text):
        html_string = BeautifulSoup(text, 'html5lib').text
        string = html_string.lower().translate(str.maketrans('', '', PUNC))
        return string

    def _process(self):
        print('\npre-precessing texts...', flush=True)
        self._clean_df = self._df.applymap(lambda x: self._parse(x))

    def is_processed(self):
        return True if self._clean_df else False

    def __iter__(self):
        if not self.is_processed:
            self._process()
        for row in self._clean_df:
            yield row

    def __str__(self):
        return '{} {} object'.format(self.status[self.is_processed()], self.__class__.__name__)

    def __len__(self):
        return self._df.shape[0]
