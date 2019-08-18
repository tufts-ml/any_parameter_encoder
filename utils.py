import math
import numpy as np



def normalize1d(logits):
    return np.array([el/sum(logits) for el in logits])


def normalize(x, axis):
    """ Normalize a 2D array
    """
    if axis == 1:
        return x / np.linalg.norm(x, axis=1, ord=1)[:, np.newaxis]
    elif axis == 0:
        return x / np.linalg.norm(x, axis=0, ord=1)
    else:
        raise ValueError('Only supports 2D normalization')


def softmax(x):
    """Note: x should be a 2D array"""
    return np.exp(x) / np.sum(np.exp(x), axis=1)[:,None]


def inverse_softmax(x):
	eps = 1e-10
	return np.log(x + eps)


def make_square(image):
    vocab_size = image.shape[1]
    nearest_square = int(math.pow(math.ceil(math.sqrt(vocab_size)), 2))
    pad_len = nearest_square - vocab_size
    if pad_len % 2 == 0:
        left_pad = pad_len / 2
        right_pad = pad_len / 2
    else:
        left_pad = pad_len / 2
        right_pad = pad_len / 2 + 1
    return np.pad(image, (left_pad, right_pad), 'constant', constant_values=(0, 0))

def unzip_X_and_topics(X_and_topics):
    X, topics = zip(*X_and_topics)
    return np.array(X), np.array(topics)