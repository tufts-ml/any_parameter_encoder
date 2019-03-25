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


train_topics = []

# single bar topics
# bottommost
for i in range(5, 10):
    horizontal_topic = np.zeros((10, 10))
    horizontal_topic[i, :] = 1
    train_topics.append(horizontal_topic)

# leftmost
for j in range(1, 5):
    vertical_topic = np.zeros((10, 10))
    vertical_topic[:, j] = 1
    train_topics.append(vertical_topic)

# all double bars except those that cross the bottom left quadrant
# topmost
for i in range(1, 5):
    # all pairs
    for j in range(1, 10):
        topic = np.zeros((10, 10))
        topic[i, :] += 1
        topic[:, j] += 1
        train_topics.append(topic)

# rightmost
for j in range(5, 10):
    # all pairs
    for i in range(1, 10):
        topic = np.zeros((10, 10))
        topic[i, :] += 1
        topic[:, j] += 1
        train_topics.append(topic)


test_single_topics = []

# single bars in the top right quadrant
# topmost (4 bars)
for i in range(1, 5):
    horizontal_topic = np.zeros((10, 10))
    horizontal_topic[i, :] = 1
    test_single_topics.append(horizontal_topic)

# rightmost (5 bars)
for j in range(6, 10):
    vertical_topic = np.zeros((10, 10))
    vertical_topic[:, j] = 1
    test_single_topics.append(vertical_topic)


test_double_topics = []
# double bars in the bottom left quadrant
# bottommost
for i in range(5, 10):
    # leftmost
    for i in range(1, 5):
        topic = np.zeros((10, 10))
        topic[i, :] += 1
        topic[:, j] += 1
        test_double_topics.append(topic)

# horizontal two-bar
for i in range(1, 9):
    topic = np.zeros((10, 10))
    topic[i, :] += 1
    topic[i + 1, :] += 1
    test_double_topics.append(topic)

# vertical two-bar
for i in range(1, 9):
    topic = np.zeros((10, 10))
    topic[:, i: i +2] += 1
    test_double_topics.append(topic)


test_triple_topics = []

# a few 3-combo topics
# all three vertical
for i in range(1, 8):
    topic = np.zeros((10, 10))
    topic[i: i + 3, :] += 1
    test_triple_topics.append(topic)

# all three horizontal
for i in range(1, 8):
    topic = np.zeros((10, 10))
    topic[:, i: i + 3] += 1
    test_triple_topics.append(topic)

# a mix
def mix(i, j, k):
    topic = np.zeros((10, 10))
    topic[:, i] += 1
    topic[j, :] += 1
    topic[:, k] += 1
    test_triple_topics.append(topic)

    topic = np.zeros((10, 10))
    topic[i, :] += 1
    topic[:, j] += 1
    topic[k, :] += 1
    test_triple_topics.append(topic)

for i in range(1, 7):
    mix(i, i + 1, i + 2)

# all 3-combo topics
# for a, b, c in combinations(range(1, 7), 3):
#     topic = np.zeros((10, 10))
#     for i in [a, b, c]:
#         is_col, index = divmod(i, 3)
#         if is_col:
#             topic[:, index] = 1`
#         else:
#             topic[index, :] = 1
#     testing_topics.append(topic)

# all 4-combo topics
# for a, b, c, d in combinations(range(1, 7), 30):
#     topic = np.zeros((10, 10))
#     for i in [a, b, c, d]:
#         is_col, index = divmod(i, 3)
#         if is_col:
#             topic[:, index] = 1
#         else:
#             topic[index, :] = 1
#     testing_topics.append(topic)