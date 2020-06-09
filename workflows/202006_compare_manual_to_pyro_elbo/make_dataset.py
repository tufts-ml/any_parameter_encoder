from dataset import create_toy_bar_docs


if __name__ == '__main__':
    kwargs = dict(
        n_topics = 10,
        vocab_size = 25,
        num_docs = 1000,
        avg_num_words = 500,
        seed = 42,
        )

    docs, true_pi_DK, true_topics_KV, true_beta_KV = create_toy_bar_docs(
        "../../data/toy_bar_docs.npy",
        **kwargs)

