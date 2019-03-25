import argparse
import os

from bnpy.viz.BarsViz import show_square_images
import numpy as np
from make_topics import train_topics, test_single_topics, test_double_topics, test_triple_topics
import matplotlib.pyplot as plt

def create_doc(topic, min_n_words_per_doc=45, max_n_words_per_doc=60, d=0):
    prng = np.random.RandomState(d)
    V = topic.size
    n_words = prng.randint(low=min_n_words_per_doc, high=max_n_words_per_doc)
    words = prng.choice(V, p=topic.flatten(), replace=True, size=n_words)
    return words


for t in train_topics:
    t /= t.sum()

test_topics = test_single_topics + test_double_topics + test_triple_topics

for t in test_topics:
    t /= t.sum()

train = [create_doc(t, d=d) for t in train_topics for d in range(100000/len(train_topics))]
valid = [create_doc(t, d=d) for t in train_topics for d in range(1000/len(train_topics))]
test = [create_doc(t, d=d) for t in test_topics for d in range(1000/len(test_topics))]

filepath = os.path.dirname(__file__)
train_filepath = os.path.join(filepath, "train.txt.npy")
valid_filepath = os.path.join(filepath, "valid.txt.npy")
test_filepath = os.path.join(filepath, "test.txt.npy")
np.save(train_filepath, train)
np.save(valid_filepath, valid)
np.save(test_filepath, test)


def plot_topics(topics, name):
    show_square_images(
        np.array(topics).reshape((len(topics), 100)) * 10, vmin=0, vmax=1, max_n_images=150)
    plt.tight_layout()
    plt.savefig(name)


plot_topics(train_topics, os.path.join(filepath, "train.png"))
plot_topics(test_topics, os.path.join(filepath, "test.png"))