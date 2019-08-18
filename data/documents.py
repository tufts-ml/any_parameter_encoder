import numpy as np
import matplotlib.pyplot as plt
from data.topics import toy_bars
from visualization.reconstructions import plot_side_by_side_docs
from utils import normalize1d


def generate_documents(topics, num_docs, alpha=.05, seed=0):
    np.random.seed(seed)
    doc_topic_dists = np.random.dirichlet([alpha] * 18, size=num_docs)
    documents = []
    for pi in doc_topic_dists:
        num_words = np.random.poisson(50)
        # word_probs = normalize1d(np.mean([a * np.array(b) for a, b in zip(pi, topics)], axis=0))
        # doc = np.random.multinomial(num_words, word_probs)
        
        doc = np.zeros(100)
        for _ in range(num_words):
            z = np.random.choice(range(18), p=pi)
            doc += np.random.multinomial(1, topics[z])
        documents.append(doc.astype(np.float32))
    return documents, doc_topic_dists


if __name__ == "__main__":
    print(len(toy_bars()))
    for alpha in [.05]:
        docs, doc_topic_dists = generate_documents(toy_bars(), 8, alpha=alpha)
        plot_side_by_side_docs(docs, name="docs_{}.png".format(alpha))
        plt.imshow(np.array(doc_topic_dists), cmap='PiYG', vmin=0, vmax=1)
        print(np.array([sorted(probs, reverse=True) for probs in doc_topic_dists]))
        plt.savefig("doc_topic_dists_{}.png".format(alpha))