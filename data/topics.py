import numpy as np
import math
from visualization.reconstructions import plot_side_by_side_docs
from utils import normalize1d


def generate_topics(n, betas, seed, shuffle=False):
    np.random.seed(seed)
    topics = []
    for beta in betas:
        topics.append(np.random.dirichlet(beta, size=n))
    topics = np.transpose(np.array(topics), [1, 0, 2])
    if shuffle:
        for topic in topics:
            np.random.shuffle(topics)
    return topics

def toy_bars(normalized=True):
    topics = []
    for i in range(1, 10):
        horizontal_topic = np.zeros(100,)
        start = 10 * i
        end = start + 10
        horizontal_topic[start: end] = 1
        if normalized:
            topics.append(normalize1d(horizontal_topic))
        else:
            topics.append(horizontal_topic)
    for i in range(1, 10):
        vertical_topic = np.zeros(100,)
        word_indices = [i + (10 * j) for j in range(10)]
        vertical_topic[word_indices] = 1
        if normalized:
            topics.append(normalize1d(vertical_topic))
        else:
            topics.append(vertical_topic)
    return topics

def remove_word_from_topic(topics, seed):
    np.random.seed(seed)
    candidate_words = []
    while len(candidate_words) <= 1:
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
    candidate_words = []
    topic_from, topic_to = np.random.randint(len(topics), size=2)
    while len(candidate_words) == 0 or topic_from == topic_to:
        topic_from, topic_to = np.random.randint(len(topics), size=2)
        candidate_words = np.intersect1d(
            np.argwhere(topics[topic_from] == 1).flatten(),
            np.argwhere(topics[topic_to] == 0).flatten()
        )
    idx_to_move = np.random.choice(candidate_words)
    topics[topic_from][idx_to_move] = 0
    topics[topic_to][idx_to_move] = 1
    return topics

def permuted_toy_bars(topics, m, seed, normalized=False):
    np.random.seed(seed)
    # topics = toy_bars(normalized=False)
    for i in range(m):
        action = np.random.randint(3)
        if action == 0:
            topics = add_word_to_topic(topics, seed+i)
        elif action == 1:
            topics = remove_word_from_topic(topics, seed+i)
        else:
            topics = move_word_between_topics(topics, seed+i)
    # deal with empty topics, if any:
    topic_word_counts = np.sum(topics, axis=1)
    for idx, c in enumerate(topic_word_counts):
        if c == 0:
            random_word = np.random.randint(100)
            topics[idx][random_word] = 1
    if normalized:
        return list(map(normalize1d, topics))
    else:
        return topics


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


def get_random_topics(n, num_topics, vocab_size, alpha, seed):
    all_topic_sets = []
    for i in range(n):
        topics = []
        for k in range(num_topics):
            topic = np.random.dirichlet([alpha] * vocab_size)
            topics.append(topic)
        all_topic_sets.append(topics)
    return all_topic_sets

def change_weights(topics):
    """ topics is a single set of topics, unnormalized"""
    new_topics = []
    for topic in topics:
        alphas = [t * 1000 + 1 for t in topic]
        new_topic = np.random.dirichlet(normalize1d(alphas), size=1)[0]
        new_topics.append(new_topic)
    return new_topics

if __name__ == "__main__":
    toy_bar_topics = toy_bars(normalized=False)
    # plot_side_by_side_docs(toy_bars(normalized=True), name="toy_bars.png")
    # for i in range(30, 300, 30):
    #     bars = np.array(change_weights(permuted_toy_bars(toy_bar_topics, i, seed=i)))
    #     plot_side_by_side_docs(bars, name="bars{}.png".format(i))
    # plot_side_by_side_docs(diagonal_bars(), name="diagonal_bars.png")
    # n_topics = 10
    # vocab_size = 100
    # # betas = .5 * np.ones((n_topics, vocab_size))
    # betas = []
    # for i in range(n_topics):
    #     beta = np.ones(vocab_size)
    #     dim = math.sqrt(vocab_size)
    #     if i < dim:
    #         popular_words = [idx for idx in range(vocab_size) if idx % dim == i]
    #     else:
    #         popular_words = [idx for idx in range(vocab_size) if int(idx / dim) == i - dim]
    #     random_additions = list(np.random.choice(range(vocab_size), 20))
    #     beta[popular_words] = 200
    #     beta[random_additions] = 50
    #     betas.append(normalize1d(beta))
    # print(betas)
    # test_topics = generate_topics(n=5, betas=betas, seed=0)
    # for i, topics in enumerate(test_topics):
    #     plot_side_by_side_docs(topics, name="test_topics{}.png".format(i))