import os
import numpy as np
from scipy.sparse import csr_matrix
from collections import Counter

from gensim.models.wrappers import LdaMallet
from gensim.models.wrappers.ldamallet import malletmodel2ldamodel


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

print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)

np.save('datasets/mdreview/train.npy', X_train)
np.save('datasets/mdreview/valid.npy', X_valid)
np.save('datasets/mdreview/test.npy', X_test)

lda_mallet = LdaMallet.load('lda_gibbs.gensim')
doc_topics = lda_mallet.load_document_topics()

train_topics = Counter()
valid_topics = Counter()
test_topics = Counter()
for i in range(X_train.shape[0]):
    topic_dist = next(doc_topics)
    topics_present = [idx for idx, pair in enumerate(topic_dist) if pair[1] > .05]
    train_topics[tuple(topics_present)] += 1

for i in range(X_valid.shape[0]):
    topic_dist = next(doc_topics)
    topics_present = [idx for idx, pair in enumerate(topic_dist) if pair[1] > .05]
    valid_topics[tuple(topics_present)] += 1

for i in range(X_test.shape[0]):
    topic_dist = next(doc_topics)
    topics_present = [idx for idx, pair in enumerate(topic_dist) if pair[1] > .05]
    test_topics[tuple(topics_present)] += 1


print(set(valid_topics) - set(train_topics))
print(set(test_topics) - set(train_topics))
#
print(train_topics)
print(valid_topics)
print(test_topics)
#
