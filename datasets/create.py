import numpy as np


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


def draw_random_doc(
        topic_list,
        min_n_words_per_doc=45,
        max_n_words_per_doc=60,
        do_return_square=True,
        proba_positive_label=0.2,
        d=0):
    prng = np.random.RandomState(d)
    V = topic_list[0].size

    # Pick which template
    # Each document is only in one topic
    k = prng.choice(len(topic_list))
    n_words = prng.randint(low=min_n_words_per_doc, high=max_n_words_per_doc)
    words = prng.choice(
        V,
        p=topic_list[k].flatten(),
        replace=True,
        size=n_words)
    return words