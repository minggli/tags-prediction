from helpers import unzip_folder, timeit
from settings import PATHS, TrainFiles
from engine import predicted_labels

@timeit
def generate_submission():

	test = unzip_folder(PATHS['DATA'], exclude=TrainFiles + ['sample_submission.csv'])[0]

	n = len(predicted_labels)
	tags = list()

	func = lambda x: ' '.join(['physics'] + list(x))

	for i in range(n):
		temp = func(predicted_labels[i])
		tags.append(temp)
		print('completed classifying {0} of {1} of test data...{2}'.format(i + 1, n, temp))

	test['tags'] = tags
	output_df = test['tags'].copy()
	output_df.to_csv(path='submission.csv', header=True, index=True, encoding='utf8')

