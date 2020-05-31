import os
import numpy as np
from utils import softmax

directory = 'data/toy_bars'
true = np.load(os.path.join(directory, 'docs_many_words_dist.npy'))[:5]
svi =  softmax(np.load(os.path.join(directory, 'svi_prior_loc.npy')))
avi =  softmax(np.load(os.path.join(directory, 'avi_template_loc.npy')))
prior =  softmax(np.load(os.path.join(directory, 'avi_prior_loc.npy')))
import ipdb; ipdb.set_trace()