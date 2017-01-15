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

nlp = pipeline()

test_string = 'James is travelling to London this Sunday. We are too.'

doc = nlp(test_string)

