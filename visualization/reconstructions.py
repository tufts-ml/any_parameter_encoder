import numpy as np
from bnpy.viz.BarsViz import show_square_images
import matplotlib.pyplot as plt

from utils import normalize
from datasets.load import load_toy_bars


def plot_side_by_side_docs(docs, name, ncols=10, intensity=10):
    """Plot recreated docs side by side (num_docs x num_methods)"""
    docs = np.asarray(docs, dtype=np.float32)
    # normalize each row so values are on a similar scale
    # (the first counts row is very different from the "recreated docs" rows)
    docs = normalize(docs, axis=1)
    vmax = np.mean(np.sum(docs, axis=1))
    show_square_images(docs * intensity, vmin=0, vmax=vmax, ncols=ncols)
    plt.tight_layout()
    plt.savefig(name)
    plt.clf()
    plt.close()


def plot_saved_samples(sample_docs, filenames, plot_name, vocab_size=100, intensity=10):
    num_examples = len(sample_docs)
    image = [sample_docs]
    for file in filenames:
        reconstructions = np.load(file)
        assert len(reconstructions) == len(sample_docs)
        image.append(reconstructions)
    num_rows = len(image)
    image = np.array(image).reshape(num_rows * num_examples, vocab_size)
    plot_side_by_side_docs(image, plot_name, ncols=num_examples, intensity=intensity)

