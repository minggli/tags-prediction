from helpers import Preprocessor, unzip_folder
from settings import PATHS, PUNC, TextMining, TrainFiles, Boundary
from engine import clf, classify, nb_test

def generate_submission():

	test = unzip_folder(PATHS['DATA'], exclude=TrainFiles + ['sample_submission.csv'])[0]

	tags = list()

	for i in range(len(nb_test)):
		temp = classify(clf=clf, word_features=nb_test[i], decision_boundary=Boundary)
		temp.insert(0, 'physics')
		temp = ' '.join(temp)
		tags.append(temp)

	test['tags'] = tags

	output_df = test['tags'].copy()
	output_df.to_csv(path='submission.csv', header=True, index=True, encoding='utf8')