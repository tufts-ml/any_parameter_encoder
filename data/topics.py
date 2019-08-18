import numpy as np
from visualization.reconstructions import plot_side_by_side_docs
from utils import normalize1d


def toy_bars():
    topics = []
    for i in range(1, 10):
        horizontal_topic = np.zeros(100,)
        start = 10 * i
        end = start + 10
        horizontal_topic[start: end] = 1
        topics.append(normalize1d(horizontal_topic))
    for i in range(1, 10):
        vertical_topic = np.zeros(100,)
        word_indices = [i + (10 * j) for j in range(10)]
        vertical_topic[word_indices] = 1
        topics.append(normalize1d(vertical_topic))
    return topics

def remove_word_from_topic(topics, seed):
    np.random.seed(seed)
    topic_idx = np.random.randint(len(topics))
    candidate_words = np.argwhere(topics[topic_idx] == 1).flatten()
    idx_to_remove = np.random.choice(candidate_words)
    topics[topic_idx][idx_to_remove] = 0
    return topics


def add_word_to_topic(topics, seed):
    np.random.seed(seed)
    topic_idx = np.random.randint(len(topics))
    candidate_words = np.argwhere(topics[topic_idx] == 0).flatten()
    idx_to_add = np.random.choice(candidate_words)
    topics[topic_idx][idx_to_add] = 1
    return topics


def move_word_between_topics(topics, seed):
    np.random.seed(seed)
    topic_from, topic_to = np.random.randint(len(topics), size=2)
    while topic_from == topic_to:
        topic_from, topic_to = np.random.randint(len(topics), size=2)
    candidate_words = np.intersect1d(
        np.argwhere(topics[topic_from] == 1).flatten(),
        np.argwhere(topics[topic_to] == 0).flatten()
    )
    idx_to_move = np.random.choice(candidate_words)
    topics[topic_from][idx_to_move] = 0
    topics[topic_to][idx_to_move] = 1
    return topics

def permuted_toy_bars(m, seed):
    np.random.seed(seed)
    topics = toy_bars()
    for i in range(m):
        action = np.random.randint(3)
        if action == 0:
            topics = add_word_to_topic(topics, seed+i)
        elif action == 1:
            topics = remove_word_from_topic(topics, seed+i)
        else:
            topics = move_word_between_topics(topics, seed+i)
    return list(map(normalize1d, topics))


def diagonal_bars():
    """ Distance from toy bars is 9 * 18 = 162
    """
    topics = []
    for j in range(1, 10):
        forward_diagonal = np.zeros((10, 10))
        i = 0
        while j >= 0:
            forward_diagonal[i, j] = 1
            i += 1
            j -= 1
        topics.append(normalize1d(forward_diagonal.flatten()))
    for i in range(1, 10):
        backward_diagonal = np.zeros((10, 10))
        j = 0
        while i < 10:
            backward_diagonal[i, j] = 1
            i += 1
            j += 1
        topics.append(normalize1d(backward_diagonal.flatten()))
    return topics

if __name__ == "__main__":
    plot_side_by_side_docs(toy_bars(), name="toy_bars.png")
    for i in range(10, 100, 10):
        bars = permuted_toy_bars(i, seed=0)
        plot_side_by_side_docs(bars, name="bars{}.png".format(i))
    plot_side_by_side_docs(diagonal_bars(), name="diagonal_bars.png")