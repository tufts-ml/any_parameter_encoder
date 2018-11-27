from bnpy.viz.BarsViz import show_square_images
import matplotlib.pyplot as plt
import numpy as np


def plot_bars(data, name):
    """ Plot documents using the toy bars plot.

    `data` should be a two-d array of the form (num_samples, num_vocabulary)

    show_square_images will infer the square structure as long as
    num_vocabulary is square

    """
    # we estimate the total counts by the mean number of words per document
    vmax = np.mean(np.sum(data, axis=1))
    show_square_images(data, vmin=0, vmax=vmax)
    plt.tight_layout()
    plt.savefig(name)