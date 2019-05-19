import os
import numpy as np
from scipy.sparse import csr_matrix
from collections import Counter
import csv

from gensim.models.wrappers import LdaMallet
from gensim.models.wrappers.ldamallet import malletmodel2ldamodel


# filepath = os.path.abspath(__file__)
# base_path = '/'.join(filepath.split('/')[:-1])
# sparse = np.load(os.path.join(base_path, 'raw/X_csr_train.npz'))
# X_train = csr_matrix((sparse['data'], sparse['indices'], sparse['indptr']), shape=sparse['shape']).todense()
# X_train = np.squeeze(np.asarray(X_train))
# vocab_size = X_train.shape[1]
#
# sparse = np.load(os.path.join(base_path, 'raw/X_csr_valid.npz'))
# X_valid = csr_matrix((sparse['data'], sparse['indices'], sparse['indptr']), shape=sparse['shape']).todense()
# X_valid = np.squeeze(np.asarray(X_valid))
#
# sparse = np.load(os.path.join(base_path, 'raw/X_csr_test.npz'))
# X_test = csr_matrix((sparse['data'], sparse['indices'], sparse['indptr']), shape=sparse['shape']).todense()
# X_test = np.squeeze(np.asarray(X_test))
#
# print(X_train.shape)
# print(X_valid.shape)
# print(X_test.shape)

X_train = np.load('datasets/mdreviews/train.npy')
X_valid = np.load('datasets/mdreviews/valid.npy')
X_test = np.load('datasets/mdreviews/test.npy')

lda_mallet = LdaMallet.load('lda_gibbs.gensim1')
doc_topics = lda_mallet.load_document_topics()
# topics = lda_mallet.get_topics()
# np.save('mdreviews_topics.npy', topics.astype(np.float32))

def np_to_id2word(X):
    corpus = []
    for doc in X:
        corpus.append([(word_idx, count) for word_idx, count in enumerate(doc)])
    return corpus

# train = np_to_id2word(X_train)
# valid = np_to_id2word(X_valid)
# test = np_to_id2word(X_test)
#
# with open('mdreviews1/results_by_sparsity1.csv', 'w') as f:
#     csv_writer = csv.writer(f)
#     for i, doc in enumerate(train):
#         topic_dist = next(doc_topics)
#         num_topics = sum([p > .05 for _, p in topic_dist])
#         csv_writer.writerow([num_topics, 'train', i])
#
#     for i, doc in enumerate(valid):
#         topic_dist = next(doc_topics)
#         num_topics = sum([p > .05 for _, p in topic_dist])
#         csv_writer.writerow([num_topics, 'valid', i])
#
#     for i, doc in enumerate(test):
#         topic_dist = next(doc_topics)
#         num_topics = sum([p > .05 for _, p in topic_dist])
#         csv_writer.writerow([num_topics, 'test', i])

# lda = malletmodel2ldamodel(lda_mallet, iterations=1000)
# with open('results_by_sparsity.csv', 'w') as f:
#     csv_writer = csv.writer(f)
#     for i, doc in enumerate(train):
#         topic_dist = lda.get_document_topics(doc)
#         num_topics = sum([p > .05 for _, p in topic_dist])
#         csv_writer.writerow([num_topics, 'train', i])
#
#     for i, doc in enumerate(valid):
#         topic_dist = lda.get_document_topics(doc)
#         num_topics = sum([p > .05 for _, p in topic_dist])
#         csv_writer.writerow([num_topics, 'valid', i])
#
#     for i, doc in enumerate(test):
#         topic_dist = lda.get_document_topics(doc)
#         num_topics = sum([p > .05 for _, p in topic_dist])
#         csv_writer.writerow([num_topics, 'test', i])

train_topics = Counter()
valid_topics = Counter()
test_topics = Counter()

for i in range(X_train.shape[0]):
    topic_dist = next(doc_topics)
    num_topics_present = len([idx for idx, pair in enumerate(topic_dist) if pair[1] > .05])
    train_topics[num_topics_present] += 1

for i in range(X_valid.shape[0]):
    topic_dist = next(doc_topics)
    num_topics_present = len([idx for idx, pair in enumerate(topic_dist) if pair[1] > .05])
    valid_topics[num_topics_present] += 1

for i in range(X_test.shape[0]):
    topic_dist = next(doc_topics)
    num_topics_present = len([idx for idx, pair in enumerate(topic_dist) if pair[1] > .05])
    test_topics[num_topics_present] += 1


# print(set(valid_topics) - set(train_topics))
# print(set(test_topics) - set(train_topics))
print(train_topics)
print(valid_topics)
print(test_topics)
print(lda_mallet.fdoctopics())