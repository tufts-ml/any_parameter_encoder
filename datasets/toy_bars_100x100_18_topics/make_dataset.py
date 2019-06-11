import argparse
import os

from bnpy.viz.BarsViz import show_square_images
import numpy as np
from make_topics import train_topics, test_single_topics, test_double_topics, test_triple_topics
import matplotlib.pyplot as plt
from datasets.create import draw_random_doc

# def resize_to_100x100(topics):
#     return [np.resize(t, (100, 100)) for t in topics]

def normalize(topics):
    for t in topics:
        t /= t.sum()
    return topics

def plot_topics(topics, name):
    show_square_images(
        np.array(topics).reshape((len(topics), 10000)) * 1000, vmin=0, vmax=1, max_n_images=150)
    plt.tight_layout()
    plt.savefig(name)


filepath = os.path.dirname(__file__)

# train_topics = resize_to_100x100(train_topics)
# test_single_topics = resize_to_100x100(test_single_topics)
# test_double_topics = resize_to_100x100(test_double_topics)
# test_triple_topics = resize_to_100x100(test_triple_topics)

plot_topics(train_topics, os.path.join(filepath, "train_before.png"))
plot_topics(test_triple_topics, os.path.join(filepath, "test_triple_before.png"))

test_topics = test_single_topics + test_double_topics + test_triple_topics

train_topics = normalize(train_topics)
test_single_topics = normalize(test_single_topics)
test_double_topics = normalize(test_double_topics)
test_triple_topics = normalize(test_triple_topics)
test_topics = normalize(test_topics)

train = [draw_random_doc(train_topics, d=d) for d in range(100000)]
valid = [draw_random_doc(train_topics, d=d + 100000) for d in range(1000)]
test = [draw_random_doc(test_topics, d=d) for d in range(1000)]
test_single = [draw_random_doc(test_single_topics, d=d) for d in range(1000)]
test_double = [draw_random_doc(test_double_topics, d=d) for d in range(1000)]
test_triple = [draw_random_doc(test_triple_topics, d=d) for d in range(1000)]

train_filepath = os.path.join(filepath, "train.txt.npy")
valid_filepath = os.path.join(filepath, "valid.txt.npy")
test_filepath = os.path.join(filepath, "test.txt.npy")
test_single_filepath = os.path.join(filepath, "test_single.txt.npy")
test_double_filepath = os.path.join(filepath, "test_double.txt.npy")
test_triple_filepath = os.path.join(filepath, "test_triple.txt.npy")
np.save(train_filepath, train)
np.save(valid_filepath, valid)
np.save(test_filepath, test)
np.save(test_single_filepath, test_single)
np.save(test_double_filepath, test_double)
np.save(test_triple_filepath, test_triple)

plot_topics(train_topics, os.path.join(filepath, "train.png"))
plot_topics(test_topics, os.path.join(filepath, "test.png"))