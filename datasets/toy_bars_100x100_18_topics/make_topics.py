import math
import numpy as np
from itertools import combinations

from utils import inverse_softmax
from datasets.create import get_bars

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
num_topics = 18
train_topics, train_single_topics, test_single_topics, test_double_topics, test_triple_topics = get_bars(vocab_size, num_topics)
single_topics = np.array(train_single_topics + test_single_topics)
print(single_topics.shape)
np.save('resources/topics_100x100_18_topics.npy', inverse_softmax(single_topics.reshape(num_topics, -1)).astype(np.float32))
    # all 3-combo topics
    # for a, b, c in combinations(range(1, 7), 3):
    #     topic = np.zeros((bar_length, bar_length))
    #     for i in [a, b, c]:
    #         is_col, index = divmod(i, 3)
    #         if is_col:
    #             topic[:, index] = 1`
    #         else:
    #             topic[index, :] = 1
    #     testing_topics.append(topic)

    # all 4-combo topics
    # for a, b, c, d in combinations(range(1, 7), 30):
    #     topic = np.zeros((bar_length, bar_length))
    #     for i in [a, b, c, d]:
    #         is_col, index = divmod(i, 3)
    #         if is_col:
    #             topic[:, index] = 1
    #         else:
    #             topic[index, :] = 1
    #     testing_topics.append(topic)

