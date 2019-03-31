import numpy as np


def normalize(x, axis):
    """ Normalize a 2D array
    """
    if axis == 1:
        return x / np.linalg.norm(x, axis=1, ord=1)[:, np.newaxis]
    elif axis == 0:
        return x / np.linalg.norm(x, axis=0, ord=1)
    else:
        raise ValueError('Only supports 2D normalization')


def inverse_softmax(x):
	eps = 1e-10
	return np.log(x + eps)