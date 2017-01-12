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

test_string = 'James is travelling to London this Sunday. We are too.'

doc = nlp(test_string)

print(doc.ents)
# testing
print(type(doc))

sentence = next(doc.sents)
print(sentence)
token = doc[1]
print(token)
sentence = next(doc.sents)
assert token is sentence[0]
