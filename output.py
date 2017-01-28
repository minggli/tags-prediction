from helpers import Preprocessor, unzip_folder
from settings import PATHS, PUNC, TextMining, TrainFiles, Boundary, Limit
from engine import classify, nb_test, train, nb_data
import os.path
import pandas as pd
import numpy as np

def generate_submission():

	clf = train()

	test = unzip_folder(PATHS['DATA'], exclude=TrainFiles + ['sample_submission.csv'])[0]

	n = len(nb_test)
	tags = list()

	func = lambda x: ' '.join(['physics'] + classify(clf=clf, word_features=x, decision_boundary=Boundary, limit=Limit))

	for i in range(n):
		temp = func(nb_test[i])
		tags.append(temp)
		print('completed classifying {0} of {1} of test data...{2}'.format(i + 1, n, temp))

	test['tags'] = tags
	output_df = test['tags'].copy()
	output_df.to_csv(path='submission.csv', header=True, index=True, encoding='utf8')
