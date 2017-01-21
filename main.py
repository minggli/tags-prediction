from helpers import CleansedDataIter, unzip_folder, test
from settings import PATHS, PUNC
import spacy

__author__ = 'Ming Li'

df = unzip_folder(PATHS['DATA'])

nlp = spacy.load('en')

texts = CleansedDataIter(df[0])


def pos_filter(doc_object, parts={'ADJ', 'DET', 'ADV', 'SPACE', 'CONJ', 'PRON', 'ADP', 'VERB', 'NOUN', 'PART'}):
    """filter unrelated parts of speech (POS) and return required parts"""
    assert isinstance(doc_object, spacy.tokens.doc.Doc), 'require a SpaCy document'
    return nlp(' '.join([str(token) for token in doc_object if token.pos_ in parts]))


def stop_word(doc_object, keep=False):
    assert isinstance(doc_object, spacy.tokens.doc.Doc), 'require a SpaCy document'
    return nlp(' '.join([str(token) for token in doc_object if token.is_stop is keep]))


def lemmatize(doc_object):
    """using SpaCy's lemmatization to performing stemming of words"""
    assert isinstance(doc_object, spacy.tokens.doc.Doc), 'require a SpaCy document'
    return nlp(' '.join([str(token.lemma_) for token in doc_object]))


def generate_training_data(data_iter, tags=False):
    for row in data_iter:
        yield row[-1] if tags is True else ' '.join(row[1:4])

# multi-treading nlp generator
for count, doc in enumerate(nlp.pipe(texts=generate_training_data(texts, tags=False), n_threads=2, batch_size=5000)):
    while count < 50:
        print(doc)
        break
