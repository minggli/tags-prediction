from helpers import Preprocessor, unzip_folder, test
from settings import PATHS, PUNC, TextMining
import spacy
import pandas as pd
import pickle

__author__ = 'Ming Li'

list_of_dataframes = unzip_folder(PATHS['DATA'], exclude=['sample_submission.csv', 'test.csv'])
df = pd.concat(objs=list_of_dataframes, ignore_index=True)
texts = Preprocessor(df)

nlp = spacy.load('en')

def pos_filter(doc_object, switch=True, parts={'ADJ', 'DET', 'ADV', 'SPACE', 'CONJ', 'PRON', 'ADP', 'VERB', 'NOUN', 'PART'}):
    """filter unrelated parts of speech (POS) and return required parts"""
    assert isinstance(doc_object, spacy.tokens.doc.Doc), 'require a SpaCy document'
    return nlp(' '.join([str(token) for token in doc_object if token.pos_ in parts])) if switch else doc_object


def stop_word(doc_object, switch=True):
    assert isinstance(doc_object, spacy.tokens.doc.Doc), 'require a SpaCy document'
    return nlp(' '.join([str(token) for token in doc_object if token.is_stop is False])) if switch else doc_object


def lemmatize(doc_object, switch=True):
    """using SpaCy's lemmatization to performing stemming of words"""
    assert isinstance(doc_object, spacy.tokens.doc.Doc), 'require a SpaCy document'
    return nlp(' '.join([str(token.lemma_) for token in doc_object])) if switch else doc_object


def pipeline(doc_object, settings={'pos': True, 'stop': True, 'lemma': True}):
    return lemmatize(stop_word(pos_filter(doc_object, switch=settings['pos']), switch=settings['stop']), switch=settings['lemma'])


def generate_training_data(data_iter, tags=False):
    for row in data_iter:
        yield row[-1] if tags is True else ' '.join(row[1:-1])

multi_threading_gen = nlp.pipe(texts=generate_training_data(texts, tags=False), batch_size=5000, n_threads=2)

data = [tuple((pipeline(feature, settings=TextMining).text, target)) for (feature, target) in
        zip(multi_threading_gen, generate_training_data(texts, tags=True))]

with open(PATHS['DATA'] + '/complete_cache.pickle', 'wb') as f:
    pickle.dump(data, f)
