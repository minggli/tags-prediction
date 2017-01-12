from utils import zip_to_df
from settings import PATHS
import pandas as pd
import numpy as np
import spacy


__author__ = 'Ming Li'

dataframes = zip_to_df(PATHS['DATA'])

# print(dataframes[3]['tags'])


def pipeline(lang='en'):
    """construct a language process pipeline"""
    return spacy.load(lang)

nlp = pipeline()

doc = nlp('we are travelling to London this Sunday.')

# testing
print(doc.ents)
