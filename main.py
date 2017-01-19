from helpers import unzip_folder, test, pipeline, CleansedData
from settings import PATHS, PUNC
import pandas as pd
import numpy as np
import spacy
from bs4 import BeautifulSoup
import string
import re


__author__ = 'Ming Li'

df = unzip_folder(PATHS['DATA'])

nlp = pipeline()

load = CleansedData(df[0])

for k, i in enumerate(load):
    if k == 13195:
        final = i
        break
    print(k, i)


# TODO find out how to predict tags

sample = [final[i] for i in range(len(final))]

doc = nlp(sample[2])

print(doc)
