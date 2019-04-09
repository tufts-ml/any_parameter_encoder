import os
import math
import numpy as np
from scipy.sparse import csr_matrix

from gensim.models import LdaModel

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

X = np.concatenate([X_train, X_valid, X_test])
corpus = []
for doc in X:
	corpus.append([(word_idx, count) for word_idx, count in enumerate(doc)])


# run SVI on the corpus
lda = LdaModel(corpus, alpha='auto')
topics = lda.get_topics()


def make_square(image):
    vocab_size = image.shape[1]
    nearest_square = math.pow(math.ceil(math.sqrt(vocab_size)), 2)
    pad_len = nearest_square - vocab_size
    if pad_len % 2 == 0:
        left_pad = pad_len / 2
        right_pad = pad_len / 2
    else:
        left_pad = pad_len / 2
        right_pad = pad_len / 2 + 1
    return np.pad(image, (left_pad, right_pad), 'constant', constant_values=(0, 0))


# run inference on a sample from the training set
# to recreate the bars
print'Inference on training set'
topic_weights, _ = lda.inference(corpus[:10])
image = [X[:10]]
image.append(np.dot(topic_weights, topics))
image = np.array(image).reshape(-1, vocab_size)
image = make_square(image)
plot_side_by_side_docs(image, 'recreated_docs_train.png')