import os
import csv
import math
import numpy as np
from scipy.sparse import csr_matrix

from gensim.models import LdaModel
from gensim.models.wrappers import LdaMallet
from utils import inverse_softmax

from utils import make_square
from visualization.reconstructions import plot_side_by_side_docs


# filepath = os.path.abspath(__file__)
# base_path = '/'.join(filepath.split('/')[:-1])
# sparse = np.load(os.path.join(base_path, 'raw/X_csr_train.npz'))
# X_train = csr_matrix((sparse['data'], sparse['indices'], sparse['indptr']), shape=sparse['shape']).todense()
# X_train = np.squeeze(np.asarray(X_train))
# vocab_size = X_train.shape[1]

# sparse = np.load(os.path.join(base_path, 'raw/X_csr_valid.npz'))
# X_valid = csr_matrix((sparse['data'], sparse['indices'], sparse['indptr']), shape=sparse['shape']).todense()
# X_valid = np.squeeze(np.asarray(X_valid))

# sparse = np.load(os.path.join(base_path, 'raw/X_csr_test.npz'))
# X_test = csr_matrix((sparse['data'], sparse['indices'], sparse['indptr']), shape=sparse['shape']).todense()
# X_test = np.squeeze(np.asarray(X_test))

# filepath = os.path.dirname(__file__)
# np.save(os.path.join(filepath, 'train.npy'), X_train)
# np.save(os.path.join(filepath, 'valid.npy'), X_valid)
# np.save(os.path.join(filepath, 'test.npy'), X_test)

# X = np.concatenate([X_train, X_valid, X_test])
vocab_size = 3000
num_topics = 30
train_docs = np.load('datasets/mdreviews/train.npy')
vocab_num_occurences = train_docs.sum(axis=0)
top_idx = np.argpartition(vocab_num_occurences, -vocab_size)[-vocab_size:]
new_train_docs = train_docs[:, top_idx]
print(new_train_docs.shape)
corpus = []
for doc in new_train_docs:
	corpus.append([(word_idx, count) for word_idx, count in enumerate(doc)])

id2word = {}
# with open('datasets/mdreviews/raw/X_colnames.txt', 'r') as f:
#     for i, line in enumerate(f):
#         id2word[i] = line.strip()
        
        

with open('datasets/mdreviews/raw/vocab.txt', 'r') as f:
    idx = 0
    for i, line in enumerate(f):
        if i in top_idx:
            id2word[idx] = line.strip()
            idx += 1
print(len(id2word))
path_to_mallet_binary = "/Users/lilyzhang/Documents/coding_projects/Mallet/bin/mallet"
with open('topics.csv', 'w') as f:
    csv_writer = csv.writer(f)
    test_topics = []
    # for i in range(1, 1001, 100):
    for i in [1000] * 10:
        lda = LdaMallet(path_to_mallet_binary, corpus=corpus, id2word=id2word, num_topics=30, iterations=i, alpha=.01)
        topic_printout = lda.show_topics(num_topics=30, num_words=10)
        for topic in topic_printout:
            topic_idx, topic_str = topic
            words = topic_str.split(' + ')
            csv_writer.writerow([i, topic_idx] + words)
        # print(lda.fdoctopics())
        # lda = LdaModel(corpus, num_topics=20, id2word=id2word)
        # print(lda.print_topics())
        test_topics.append(lda.get_topics())
    # np.save('resources/mdreviews_topics3.npy', topics)
    # lda.save('lda_gibbs.gensim3')
    np.save('datasets/mdreviews/test_topics_3k_new.npy', np.array(test_topics))
# topics = np.load('true_topics.npy')
# take the inverse softmax


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



