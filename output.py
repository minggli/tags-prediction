from helpers import Preprocessor, unzip_folder
from settings import PATHS, PUNC, TextMining, TrainFiles, Boundary, Limit
from engine import classify, nb_test, train, nb_data
import os.path

def generate_submission():

	clf = train()

	test = unzip_folder(PATHS['DATA'], exclude=TrainFiles + ['sample_submission.csv'])[0]

	tags = list()

	n = len(nb_test)

	for i in range(n):
		temp = classify(clf=clf, word_features=nb_test[i], decision_boundary=Boundary, limit=Limit)
		temp.insert(0, 'physics')
		temp = ' '.join(temp)
		tags.append(temp)
		print('completed classifying {0} of {1} of test data...'.format(i + 1, n))

	test['tags'] = tags

	output_df = test['tags'].copy()
	output_df.to_csv(path='submission.csv', header=True, index=True, encoding='utf8')