import numpy as np
from bnpy.viz.BarsViz import show_square_images
import matplotlib.pyplot as plt

from utils import normalize


def plot_side_by_side_docs(docs, name, ncols=10):
    """Plot recreated docs side by side (num_docs x num_methods)"""
    docs = np.asarray(docs, dtype=np.float32)
    # normalize each row so values are on a similar scale
    # (the first counts row is very different from the "recreated docs" rows)
    docs = normalize(docs, axis=1)
    vmax = np.mean(np.sum(docs, axis=1))
    show_square_images(docs * 10, vmin=0, vmax=vmax, ncols=ncols)
    plt.tight_layout()
    plt.savefig(name)
    plt.clf()
    plt.close()