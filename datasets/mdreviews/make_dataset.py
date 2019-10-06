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
# np.save(os.path.join(filepath, 'train.npy'), X_train)
# np.save(os.path.join(filepath, 'valid.npy'), X_valid)
# np.save(os.path.join(filepath, 'test.npy'), X_test)

X = np.concatenate([X_train, X_valid, X_test])
corpus = []
for doc in X:
	corpus.append([(word_idx, count) for word_idx, count in enumerate(doc)])

id2word = {}
# with open('datasets/mdreviews/raw/X_colnames.txt', 'r') as f:
#     for i, line in enumerate(f):
#         id2word[i] = line.strip()

a = [(0, '0.007*"book" + 0.005*"good" + 0.005*"great" + 0.005*"movie" + 0.004*"read" + 0.004*"work" + 0.004*"time" + 0.003*"i_was" + 0.003*"love" + 0.003*"make"'), (1, '0.007*"book" + 0.005*"good" + 0.005*"read" + 0.004*"great" + 0.004*"don\'t" + 0.004*"i_have" + 0.004*"time" + 0.003*"movie" + 0.003*"buy" + 0.003*"charact"'), (2, '0.008*"book" + 0.005*"great" + 0.005*"read" + 0.004*"good" + 0.004*"work" + 0.003*"i_have" + 0.003*"film" + 0.003*"time" + 0.003*"movie" + 0.003*"this_book"'), (3, '0.008*"book" + 0.004*"great" + 0.004*"movie" + 0.004*"read" + 0.004*"time" + 0.003*"if_you" + 0.003*"film" + 0.003*"good" + 0.003*"i_have" + 0.003*"work"'), (4, '0.008*"book" + 0.005*"time" + 0.005*"great" + 0.004*"read" + 0.004*"movie" + 0.004*"don\'t" + 0.004*"good" + 0.003*"work" + 0.003*"make" + 0.003*"this_book"'), (5, '0.008*"book" + 0.004*"good" + 0.004*"great" + 0.004*"movie" + 0.003*"read" + 0.003*"work" + 0.003*"if_you" + 0.003*"make" + 0.003*"this_book" + 0.003*"don\'t"'), (6, '0.007*"book" + 0.005*"great" + 0.005*"good" + 0.005*"read" + 0.004*"movie" + 0.004*"work" + 0.004*"this_book" + 0.004*"time" + 0.004*"don\'t" + 0.003*"i_have"'), (7, '0.007*"book" + 0.005*"good" + 0.004*"read" + 0.004*"movie" + 0.004*"great" + 0.004*"if_you" + 0.003*"work" + 0.003*"thing" + 0.003*"i_have" + 0.003*"time"'), (8, '0.006*"book" + 0.005*"movie" + 0.005*"good" + 0.005*"time" + 0.004*"great" + 0.004*"don\'t" + 0.003*"read" + 0.003*"work" + 0.003*"film" + 0.003*"this_book"'), (9, '0.008*"book" + 0.005*"great" + 0.004*"good" + 0.004*"read" + 0.004*"time" + 0.003*"movie" + 0.003*"film" + 0.003*"work" + 0.003*"this_book" + 0.003*"thing"'), (10, '0.006*"book" + 0.005*"good" + 0.004*"read" + 0.004*"don\'t" + 0.004*"time" + 0.004*"great" + 0.004*"movie" + 0.003*"i_have" + 0.003*"buy" + 0.003*"story"'), (11, '0.006*"book" + 0.005*"good" + 0.004*"movie" + 0.004*"great" + 0.004*"read" + 0.003*"this_book" + 0.003*"time" + 0.003*"i_have" + 0.003*"film" + 0.003*"work"'), (12, '0.006*"book" + 0.004*"good" + 0.004*"great" + 0.004*"work" + 0.004*"movie" + 0.004*"time" + 0.003*"film" + 0.003*"i_have" + 0.003*"this_book" + 0.003*"if_you"'), (13, '0.007*"book" + 0.005*"great" + 0.004*"good" + 0.004*"movie" + 0.004*"film" + 0.003*"time" + 0.003*"work" + 0.003*"read" + 0.003*"product" + 0.003*"i_have"'), (14, '0.006*"book" + 0.004*"make" + 0.004*"good" + 0.004*"work" + 0.004*"time" + 0.004*"great" + 0.003*"don\'t" + 0.003*"film" + 0.003*"read" + 0.003*"charact"'), (15, '0.006*"book" + 0.005*"great" + 0.005*"good" + 0.005*"movie" + 0.004*"don\'t" + 0.004*"time" + 0.003*"film" + 0.003*"i\'m" + 0.003*"read" + 0.003*"thing"'), (16, '0.008*"book" + 0.005*"great" + 0.004*"work" + 0.004*"good" + 0.004*"read" + 0.003*"movie" + 0.003*"this_book" + 0.003*"i_have" + 0.003*"and_i" + 0.003*"make"'), (17, '0.008*"book" + 0.005*"read" + 0.005*"work" + 0.005*"great" + 0.004*"if_you" + 0.004*"movie" + 0.003*"good" + 0.003*"film" + 0.003*"time" + 0.003*"product"'), (18, '0.008*"book" + 0.005*"good" + 0.005*"great" + 0.005*"time" + 0.004*"read" + 0.004*"work" + 0.004*"movie" + 0.003*"this_book" + 0.003*"film" + 0.003*"product"'), (19, '0.008*"book" + 0.006*"good" + 0.004*"movie" + 0.004*"great" + 0.004*"this_book" + 0.004*"work" + 0.003*"product" + 0.003*"read" + 0.003*"make" + 0.003*"buy"'), (20, '0.008*"book" + 0.004*"good" + 0.004*"work" + 0.004*"great" + 0.004*"read" + 0.004*"movie" + 0.004*"time" + 0.003*"don\'t" + 0.003*"make" + 0.003*"i_have"'), (21, '0.006*"book" + 0.005*"great" + 0.004*"movie" + 0.004*"work" + 0.003*"good" + 0.003*"i_have" + 0.003*"don\'t" + 0.003*"make" + 0.003*"time" + 0.003*"product"'), (22, '0.007*"book" + 0.006*"good" + 0.004*"great" + 0.004*"movie" + 0.004*"time" + 0.004*"work" + 0.003*"read" + 0.003*"film" + 0.003*"buy" + 0.003*"this_book"'), (23, '0.007*"book" + 0.005*"time" + 0.005*"good" + 0.004*"read" + 0.004*"movie" + 0.003*"don\'t" + 0.003*"great" + 0.003*"work" + 0.003*"i_have" + 0.003*"problem"'), (24, '0.008*"book" + 0.005*"great" + 0.005*"movie" + 0.004*"good" + 0.004*"time" + 0.004*"film" + 0.004*"read" + 0.003*"this_book" + 0.003*"i_have" + 0.003*"thing"'), (25, '0.008*"book" + 0.005*"great" + 0.004*"don\'t" + 0.004*"i_have" + 0.004*"movie" + 0.004*"good" + 0.004*"work" + 0.004*"make" + 0.003*"read" + 0.003*"time"'), (26, '0.008*"book" + 0.005*"great" + 0.005*"good" + 0.004*"movie" + 0.004*"work" + 0.003*"film" + 0.003*"time" + 0.003*"don\'t" + 0.003*"i_have" + 0.003*"this_book"'), (27, '0.006*"book" + 0.005*"good" + 0.005*"great" + 0.004*"movie" + 0.004*"read" + 0.003*"time" + 0.003*"i_have" + 0.003*"review" + 0.003*"don\'t" + 0.003*"work"'), (28, '0.007*"book" + 0.004*"read" + 0.004*"great" + 0.004*"time" + 0.004*"don\'t" + 0.004*"good" + 0.003*"work" + 0.003*"movie" + 0.003*"i_have" + 0.003*"review"'), (29, '0.008*"book" + 0.005*"movie" + 0.005*"great" + 0.004*"good" + 0.003*"time" + 0.003*"if_you" + 0.003*"this_book" + 0.003*"don\'t" + 0.003*"i_have" + 0.003*"read"')]
        
        

with open('datasets/mdreviews/raw/vocab.txt', 'r') as f:
    for i, line in enumerate(f):
        id2word[i] = line.strip()
path_to_mallet_binary = "/Users/lilyzhang/Documents/coding_projects/Mallet/bin/mallet"
with open('topics.csv', 'w') as f:
    csv_writer = csv.writer(f)
    test_topics = []
    for i in range(1, 1001, 100):
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
    np.save('datasets/mdreviews/test_topics_3k.npy', np.array(test_topics))
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



