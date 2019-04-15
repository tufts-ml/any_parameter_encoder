import os
import math
import numpy as np
from scipy.sparse import csr_matrix

from gensim.models import LdaModel
from gensim.models.wrappers import LdaMallet

from utils import make_square
from visualization.reconstructions import plot_side_by_side_docs


filepath = os.path.abspath(__file__)
base_path = '/'.join(filepath.split('/')[:-1])
sparse = np.load(os.path.join(base_path, 'raw/X_csr_train.npz'))
X_train = csr_matrix((sparse['data'], sparse['indices'], sparse['indptr']), shape=sparse['shape']).todense()
X_train = np.squeeze(np.asarray(X_train))
vocab_size = X_train.shape[1]

sparse = np.load(os.path.join(base_path, 'raw/X_csr_valid.npz'))
X_valid = csr_matrix((sparse['data'], sparse['indices'], sparse['indptr']), shape=sparse['shape']).todense()
X_valid = np.squeeze(np.asarray(X_valid))

sparse = np.load(os.path.join(base_path, 'raw/X_csr_test.npz'))
X_test = csr_matrix((sparse['data'], sparse['indices'], sparse['indptr']), shape=sparse['shape']).todense()
X_test = np.squeeze(np.asarray(X_test))

filepath = os.path.dirname(__file__)
np.save(os.path.join(filepath, 'train.npy'), X_train)
np.save(os.path.join(filepath, 'valid.npy'), X_valid)
np.save(os.path.join(filepath, 'test.npy'), X_test)

# X = np.concatenate([X_train, X_valid, X_test])
# corpus = []
# for doc in X:
# 	corpus.append([(word_idx, count) for word_idx, count in enumerate(doc)])


# run SVI on the corpus
# lda = LdaModel(corpus, num_topics=20, alpha='auto')
# topics = lda.get_topics()
# np.save('true_topics.npy', topics)
# lda.save('lda.gensim')

# path_to_mallet_binary = "/Users/lilyzhang/Documents/coding_projects/Mallet/bin/mallet"
# lda = LdaMallet(path_to_mallet_binary, corpus=corpus, num_topics=20)
# topics = lda.get_topics()
# np.save('true_topics.npy', topics)
# lda.save('lda_gibbs.gensim')

# lda = LdaModel.load('lda.gensim')
# topics = lda.get_topics()

# run inference on a sample from the training set
# to recreate the bars
# print'Inference on training set'
# topic_weights = []
# for doc in corpus[:1000]:
# 	topic_weights.append(lda.get_document_topics(doc))


# topic_weights, _ = lda.inference(corpus[:10])
# np.save('doc_topic_weights.npy', np.array(topic_weights))
# reconstruction = np.dot(topic_weights, topics)



