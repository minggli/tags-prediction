from helpers import unzip_folder, test, pipeline, CleansedData
from settings import PATHS, PUNC
import pandas as pd
import numpy as np
import spacy


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

doc = nlp(str(sample[2]))

print(doc)

for token in doc:
    print(token.pos_, token.pos)
