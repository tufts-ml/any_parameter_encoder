import numpy as np


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