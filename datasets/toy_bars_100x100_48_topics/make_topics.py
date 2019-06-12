import numpy as np
from datasets.toy_bars_100x100_18_topics.make_topics import get_bars

from utils import inverse_softmax


# top and leftmost bars never used
# train: half of single topics (8, 5 horizontal and 4 vertical),
# the leftmost and the bottom most
# all double bars except those that cross the bottom left quadrant,
# i.e. the topmost and rightmost bars

# test1: single bars in the top right quadrant
# test2: double bars in the bottom left quadrant and consecutive double bars
# (single bars exist, but never these specific combinations)
# test3: triple bars
vocab_size = 10000
num_topics = 48
train_topics, train_single_topics, test_single_topics, test_double_topics, test_triple_topics = get_bars(vocab_size, num_topics)
single_topics = np.array(train_single_topics + test_single_topics)
print(single_topics.shape)
np.save('resources/topics_100x100_48_topics.npy', inverse_softmax(single_topics.reshape(num_topics, -1)).astype(np.float32))