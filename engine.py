from settings import PATHS
from nltk import NaiveBayesClassifier
import pickle
from gensim import models, corpora
from helpers import word_feat

with open(PATHS['DATA'] + '/cache.pickle', 'rb') as f:
	data = pickle.load(f)

# trying Latent Latent Dirichlet Allocation (LDA)



nb_train = [i for i in map(lambda x: tuple((word_feat(x[0].split(), numeric=False), x[1])), data[:20])]

print(nb_train[3])

clf = NaiveBayesClassifier.train(nb_train)




