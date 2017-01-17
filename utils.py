import pandas as pd
import os
import zipfile as zf
import spacy
import re
from bs4 import BeautifulSoup

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


class Parser(object):

    def __init__(self, df, titles=['title', 'content', 'tags']):
        assert isinstance(df, pd.DataFrame), 'input require Pandas DataFrame object.'
        self._df = df
        self._titles = titles if titles else df.columns

