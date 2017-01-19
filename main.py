from helpers import unzip_folder, test, pipeline, CleansedData
from settings import PATHS, PUNC
import pandas as pd
import numpy as np
import spacy


__author__ = 'Ming Li'

df = unzip_folder(PATHS['DATA'])

nlp = pipeline()

load = CleansedData(df[-1])

for k, i in enumerate(load):
    if k == 13195:
        final = i
        break

# TODO find out how to predict tags

sample = [final[i] for i in range(len(final))]

doc = nlp(str(sample[1]))
print(load.is_processed())


def pos_filter(doc_object, parts=['NOUN']):
    """filter unrelated parts of speech (POS) and return required parts"""
    assert isinstance(doc_object, spacy.tokens.doc.Doc), 'require a SpaCy document'
    return nlp(' '.join([str(token) for token in doc_object if token.pos_ in parts]))


def lemmatize(doc_object):
    assert isinstance(doc_object, spacy.tokens.doc.Doc), 'require a SpaCy document'
    return nlp(' '.join([str(token.lemma_) for token in doc_object]))

print(doc)

print(pos_filter(lemmatize(doc), parts=['NOUN']))

