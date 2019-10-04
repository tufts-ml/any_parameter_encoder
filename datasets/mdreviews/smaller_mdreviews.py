import numpy as np
from gensim.models.wrappers import LdaMallet


vocab_size = 3000
num_topics = 30
train_docs = np.load('datasets/mdreviews/train.npy')
vocab_num_occurences = train_docs.sum(axis=0)
top_idx = np.argpartition(vocab_num_occurences, -vocab_size)[-vocab_size:]
new_train_docs = train_docs[:, top_idx]
print(new_train_docs.shape)

id2word = {}
words = []
with open('datasets/mdreviews/raw/vocab.txt', 'r') as f:
    for line in f:
        words.append(line.strip())
new_words = np.array(words)[top_idx]
print(new_words.shape)
id2word = {i: word for i, word in enumerate(new_words)}
path_to_mallet_binary = "/Users/lilyzhang/Documents/coding_projects/Mallet/bin/mallet"
corpus = []
for doc in new_train_docs:
	corpus.append([(word_idx, count) for word_idx, count in enumerate(doc)])
lda = LdaMallet(path_to_mallet_binary, corpus=corpus, id2word=id2word, num_topics=num_topics, iterations=1000, alpha=.01)
print(lda.print_topics())
topics = lda.get_topics()
np.save('resources/mdreviews_topics4.npy', topics)
lda.save('lda_gibbs.gensim4')

