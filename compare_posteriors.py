import os
import numpy as np
from utils import softmax


directory = 'data/toy_bars'
true = np.load(os.path.join(directory, 'docs_many_words_dist.npy'))[:5]
svi =  softmax(np.load(os.path.join(directory, 'svi_prior_loc.npy')))
avi =  softmax(np.load(os.path.join(directory, 'avi_template_loc.npy')))
prior =  softmax(np.load(os.path.join(directory, 'avi_prior_loc.npy')))
docs = np.load(os.path.join(directory, 'docs_many_words.npy'))[:5]
topics = np.load(os.path.join(directory, 'topics_many_words.npy'))
k = np.matmul(docs, np.transpose(topics, (0, 2, 1)))
i = 0
def get_top_n(idx):
    print('true', np.argpartition(true[idx], 4)[-4:])
    print('svi', np.argpartition(svi[idx], 4)[-4:])
    print('avi', np.argpartition(avi[idx], 4)[-4:])
    print('k', np.argpartition(k[idx], 4)[-4:])
print(get_top_n(i))
import ipdb; ipdb.set_trace()