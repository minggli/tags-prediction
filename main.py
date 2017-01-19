from helpers import CleansedDataIter, unzip_folder, test
from settings import PATHS, PUNC
import pandas as pd
import numpy as np
import spacy

__author__ = 'Ming Li'

df = unzip_folder(PATHS['DATA'])

nlp = spacy.en.English(tagger=True, parser=True, entity=True)

texts = CleansedDataIter(df[0])

for k, i in enumerate(texts):
    if k == 13195:
        final = i
        break

sample = final[1:4]


def pos_filter(doc_object, parts={'ADJ', 'DET', 'ADV', 'SPACE', 'CONJ', 'PRON', 'ADP', 'VERB', 'NOUN', 'PART'}
               , stop_word=False):
    """filter unrelated parts of speech (POS) and return required parts"""
    assert isinstance(doc_object, spacy.tokens.doc.Doc), 'require a SpaCy document'
    return nlp(' '.join([str(token) for token in doc_object if token.pos_ in parts and token.is_stop is stop_word]))


def lemmatize(doc_object):
    """using SpaCy's lemmatization to performing stemming of words"""
    assert isinstance(doc_object, spacy.tokens.doc.Doc), 'require a SpaCy document'
    return nlp(' '.join([str(token.lemma_) for token in doc_object]))

doc = nlp.make_doc(' '.join(sample))
print(type(doc))

# combined = pos_filter(lemmatize(doc), stop_word=False)
#
# print(doc)
# print(combined)


for doc in nlp.pipe(texts=texts, n_threads=3, batch_size=10000):
    print(len(doc))

