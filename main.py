from utils import unzip_folder, test, pipeline, Parser
from settings import PATHS
import pandas as pd
import numpy as np
import spacy
from bs4 import BeautifulSoup


__author__ = 'Ming Li'

dataframes = unzip_folder(PATHS['DATA'])

# nlp = pipeline()

# TODO find out how to predict tags

#
# doc = nlp(sample['content'])

sample = dataframes[0]['content'].loc[51248]

# print(sample)

soup = BeautifulSoup(sample, 'html5lib')

print(dataframes[0]['content'])
print(soup.text)

