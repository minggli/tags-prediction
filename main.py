from utils import zip_to_df
from settings import PATHS
import pandas as pd
import numpy as np
import spacy


__author__ = 'Ming Li'

dataframes = zip_to_df(PATHS['DATA'])

print(dataframes[-2])

