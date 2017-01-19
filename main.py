from utils import unzip_folder, test, pipeline, CleansedData
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


load = CleansedData(df[0])

for k, i in enumerate(load):
    if k == 13195:
        final = i
    print(i)

print(final[2])

