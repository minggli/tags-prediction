from utils import unzip_folder, test, pipeline, Cleanse
from settings import PATHS, PUNC
import pandas as pd
import numpy as np
import spacy
from bs4 import BeautifulSoup
import string
import re


__author__ = 'Ming Li'

df = unzip_folder(PATHS['DATA'])

# nlp = pipeline()

# TODO find out how to predict tags

# sample = sample.translate(sample.maketrans('', '', PUNC))

# doc = nlp(sample)

# print(soup.text)


load = Cleanse(df[0])
print(len(load))
# for i in load:
#     print(i)