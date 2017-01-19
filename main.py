from helpers import CleansedData, unzip_folder, test, pipeline
from settings import PATHS, PUNC
import pandas as pd
import numpy as np
import spacy

__author__ = 'Ming Li'

df = unzip_folder(PATHS['DATA'])

nlp = pipeline()

load = CleansedData(df[1])

for k, i in enumerate(load):
    if k == 13195:
        final = i
        break

sample = [str(final[i]) for i in range(len(final))]


def pos_filter(doc_object, parts={'ADJ', 'DET', 'ADV', 'SPACE', 'CONJ', 'PRON', 'ADP', 'VERB', 'NOUN', 'PART'}
           , stop_word=False):
    """filter unrelated parts of speech (POS) and return required parts"""
    assert isinstance(doc_object, spacy.tokens.doc.Doc), 'require a SpaCy document'
    return nlp(' '.join([str(token) for token in doc_object if token.pos_ in parts and token.is_stop is stop_word]))


def lemmatize(doc_object):
    """using SpaCy's lemmatization to standardize words"""
    assert isinstance(doc_object, spacy.tokens.doc.Doc), 'require a SpaCy document'
    return nlp(' '.join([str(token.lemma_) for token in doc_object]))

doc1 = nlp(str(sample[1]))
doc2 = nlp(str(sample[2]))

combined = pos_filter(lemmatize(doc1), stop_word=False) + pos_filter(lemmatize(doc2), stop_word=False)

print(doc1 + doc2)
print(combined)
