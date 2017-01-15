from utils import unzip_folder
from settings import PATHS
import pandas as pd
import numpy as np
import spacy


__author__ = 'Ming Li'

dataframes = unzip_folder(PATHS['DATA'])


def pipeline(lang='en'):
    """construct a language process pipeline"""
    return spacy.load(lang)


def test(test_string='James is travelling to London this Sunday. We are too.'):

    doc = nlp(test_string)

    for sentence in doc.sents:
        print(sentence)

    for token in doc:
        print(token, token.tag, token.tag_, token.lemma, token.lemma_, token.pos, token.pos_)

    for ent in doc.ents:
        print(ent, ent.label, ent.label_)

    for np in doc.noun_chunks:
        print(np)

    print(doc[0].similarity(doc[6]))


nlp = pipeline()
test()

# TODO find out how to predict tags
