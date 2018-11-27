import numpy as np
from gensim.models import LdaModel
from metrics import plots


# This script uses online variation bayes as inference

dataset_tr = 'datasets/toy_bars/train.txt.npy'
data_tr = np.load(dataset_tr)
# only use 5000
# data_tr = data_tr[:5000]
dataset_te = 'datasets/toy_bars/test.txt.npy'
data_te = np.load(dataset_te)
vocab_size = 9
#--------------convert to one-hot representation------------------
print 'Loaded data'
counts_tr = np.array(
	[np.bincount(doc.astype('int'), minlength=vocab_size)
	 for doc in data_tr if np.sum(doc)!=0]
)
counts_te = np.array(
	[np.bincount(doc.astype('int'), minlength=vocab_size)
	 for doc in data_te if np.sum(doc)!=0]
)

corpus_tr = []
corpus_te = []
for doc_tr in counts_tr:
	corpus_tr.append([(word_idx, count) for word_idx, count in enumerate(doc_tr)])

for doc_te in counts_te:
	corpus_te.append([(word_idx, count) for word_idx, count in enumerate(doc_te)])

assert len(counts_tr) == len(data_tr)
assert len(counts_te) == len(data_te)
assert len(corpus_tr) == len(data_tr)
assert len(corpus_te) == len(data_te)

print'Created corpus'

lda = LdaModel(corpus_tr, num_topics=10)
topics = lda.get_topics()
plots.plot_bars(topics, 'inferred_topics_ovb.png')

# run inference on a sample from the training set
# to recreate the bars
print'Inference on training set'
topic_weights, _ = lda.inference(corpus_tr[:12])
recreated_tr = np.dot(topic_weights, topics)
plots.plot_bars(recreated_tr, 'recreated_docs_train_ovb.png')
# run inference on a sample from the test set
# to recreate the bars without updating the params
print'Inference on test set'
topic_weights, _ = lda.inference(corpus_te[:12])
recreated_te = np.dot(topic_weights, topics)
plots.plot_bars(recreated_te, 'recreated_docs_test_ovb.png')

# run inference on a sample from the test set
# to recreate the bars after updating the params
print'Model update and inference on test set'
lda.update(corpus_te)
topic_weights, _ = lda.inference(corpus_te[:12])
recreated_te1 = np.dot(topic_weights, topics)
plots.plot_bars(recreated_te1, 'recreated_docs_test_ovb_after_inference.png')

