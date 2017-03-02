from helpers import Preprocessor, unzip_folder
from settings import PATHS, TrainFiles
import numpy as np
import os
import pandas as pd
import sys

__author__ = 'Ming Li'

EVAL = True if 'EVAL' in map(str.upper, sys.argv[1:]) else False

list_of_dataframes = unzip_folder(PATHS['DATA'], exclude=['sample_submission.csv', 'test.csv'])
df = pd.concat(objs=list_of_dataframes, ignore_index=True)
train_iter = Preprocessor(df)

test = unzip_folder(PATHS['DATA'], exclude=TrainFiles + ['sample_submission.csv'])[0]
test_iter = Preprocessor(test)

if not os.path.exists(PATHS['DATA'] + '/train_features.npy') or not os.path.exists(PATHS['DATA'] + '/train_labels.npy'):

    train_features, train_labels = list(), list()
    for feat, label in train_iter:
        train_features.append(feat)
        train_labels.append(label)

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)

    with open(PATHS['DATA'] + '/train_features.npy', 'wb') as f:
        train_features.dump(f)

    with open(PATHS['DATA'] + '/train_labels.npy', 'wb') as f:
        train_labels.dump(f)


if not os.path.exists(PATHS['DATA'] + '/test_features.npy'):

    test_features = list()
    for feat, _ in test_iter:
        test_features.append(feat)

    with open(PATHS['DATA'] + '/test_features.npy', 'wb') as f:
        test_features.dump(f)


if __name__ == '__main__':

    if EVAL:
        from output import generate_submission
        generate_submission()
