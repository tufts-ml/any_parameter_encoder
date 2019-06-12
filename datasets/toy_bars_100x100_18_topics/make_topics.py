import math
import numpy as np
from itertools import combinations

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

def get_bars(vocab_size, num_topics):
    bar_length = int(math.sqrt(vocab_size))
    bar_width = bar_length / (num_topics / 2 + 1)
    print('bar_length', bar_length)
    print('bar_width', bar_width)
    train_topics = []
    train_single_topics = []
    min_idx = 1
    max_idx = num_topics / 2 + 1
    mid_idx = max_idx / 2
    # single bar topics
    # bottommost
    for i in range(mid_idx, max_idx):
        horizontal_topic = np.zeros((bar_length, bar_length))
        horizontal_topic[i * bar_width: bar_width + i * bar_width, :] = 1
        train_topics.append(horizontal_topic)
        train_single_topics.append(horizontal_topic)

    # leftmost
    for j in range(1, mid_idx):
        vertical_topic = np.zeros((bar_length, bar_length))
        vertical_topic[:, j * bar_width: bar_width + j* bar_width] = 1
        train_topics.append(vertical_topic)
        train_single_topics.append(vertical_topic)

    # all double bars except those that cross the bottom left quadrant
    # topmost
    for i in range(1, mid_idx):
        # all pairs
        for j in range(1, max_idx):
            topic = np.zeros((bar_length, bar_length))
            topic[i * bar_width: bar_width + i * bar_width, :] += 1
            topic[:, j * bar_width: bar_width + j* bar_width] += 1
            train_topics.append(topic)

    # rightmost
    for j in range(mid_idx, max_idx):
        # all pairs
        for i in range(1, max_idx):
            topic = np.zeros((bar_length, bar_length))
            topic[i * bar_width: bar_width + i * bar_width, :] += 1
            topic[:, j * bar_width: bar_width + j* bar_width] += 1
            train_topics.append(topic)


    test_single_topics = []

    # single bars in the top right quadrant
    # topmost (4 bars)
    for i in range(1, mid_idx):
        horizontal_topic = np.zeros((bar_length, bar_length))
        horizontal_topic[i * bar_width: bar_width + i* bar_width, :] = 1
        test_single_topics.append(horizontal_topic)

    # rightmost (mid_idx bars)
    for j in range(mid_idx, max_idx):
        vertical_topic = np.zeros((bar_length, bar_length))
        vertical_topic[:, j * bar_width: bar_width + j* bar_width] = 1
        test_single_topics.append(vertical_topic)


    test_double_topics = []
    # double bars in the bottom left quadrant
    # bottommost
    for i in range(mid_idx, max_idx):
        # leftmost
        for i in range(1, mid_idx):
            topic = np.zeros((bar_length, bar_length))
            topic[i * bar_width: (bar_width + i * bar_width), :] += 1
            topic[:, j * bar_width: bar_width + j* bar_width] += 1
            test_double_topics.append(topic)

    # horizontal two-bar
    for i in range(1, max_idx - 1):
        topic = np.zeros((bar_length, bar_length))
        topic[i * bar_width: 2 * bar_width + i * bar_width, :] += 1
        test_double_topics.append(topic)

    # vertical two-bar
    for i in range(1, max_idx - 1):
        topic = np.zeros((bar_length, bar_length))
        topic[:, i * bar_width: 2 * bar_width + i * bar_width] += 1
        test_double_topics.append(topic)


    test_triple_topics = []

    # a few 3-combo topics
    # all three vertical
    for i in range(1, max_idx - 2):
        topic = np.zeros((bar_length, bar_length))
        topic[i * bar_width: 3 * bar_width + i * bar_width, :] += 1
        test_triple_topics.append(topic)

    # all three horizontal
    for i in range(1, max_idx - 2):
        topic = np.zeros((bar_length, bar_length))
        topic[:, i * bar_width: 3 * bar_width + i * bar_width] += 1
        test_triple_topics.append(topic)

    # a mix
    def mix(i, j, k):
        topic = np.zeros((bar_length, bar_length))
        topic[:, i * bar_width: bar_width + i * bar_width] += 1
        topic[j * bar_width: bar_width + j* bar_width, :] += 1
        topic[:, k * bar_width: bar_width + k* bar_width] += 1
        test_triple_topics.append(topic)

        topic = np.zeros((bar_length, bar_length))
        topic[i * bar_width: bar_width + i * bar_width, :] += 1
        topic[:, j * bar_width: bar_width + j* bar_width] += 1
        topic[k * bar_width: bar_width + k* bar_width, :] += 1
        test_triple_topics.append(topic)

    for i in range(1, max_idx - 3):
        mix(i, i + 1, i + 2)

    return train_topics, train_single_topics, test_single_topics, test_double_topics, test_triple_topics
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

