import math
import numpy as np
import torch
from pyro.distributions.util import logsumexp
from bnpy.viz.BarsViz import show_square_images
import matplotlib.pyplot as plt


def evaluate_log_predictive_density(posterior_predictive_traces):
    """
    Evaluate the log probability density of observing the unseen data
    given a model and empirical distribution over the parameters.
    """
    trace_log_pdf = []
    for tr in posterior_predictive_traces.exec_traces:
        trace_log_pdf.append(tr.log_prob_sum())
    # Use LogSumExp trick to evaluate $log(1/num_samples \sum_i p(new_data | \theta^{i})) $,
    # where $\theta^{i}$ are parameter samples from the model's posterior.
    posterior_pred_density = logsumexp(torch.stack(trace_log_pdf), dim=-1) - math.log(len(trace_log_pdf))
    print("\nLog posterior predictive density")
    print("--------------------------------")
    print("{:.4f}\n".format(posterior_pred_density))
    return posterior_pred_density


def reconstruct_data_map(posterior_predictive_traces, inference, vae, x):
    """Generate data from traces
    """
    if inference == 'vae':
        z_loc, _ = vae.encoder(x)
    if inference == 'svi':
        pass  #TODO; get the learned params
    if inference == 'mcmc':
        pass  # TODO; get MAP from traces
    # decode the image (note we don't sample in image space)
    word_probs = vae.decoder(z_loc)
    return word_probs


def reconstruct_data(posterior, vae):
    """Generate data from traces
    """
    # encode image x
    reconstructions = []
    # each tr contains a trace for every datapoint
    for tr in posterior.exec_traces:
        z_loc = tr.nodes['latent']['value']
        # decode the image (note we don't sample in image space)
        reconstruction = vae.decoder(z_loc).detach().numpy()
        reconstructions.append(reconstruction)
    return np.array(reconstructions)
    # we want each set of reconstructions to be a column
    # reconstructions = zip(*reconstructions)
    # return np.array(reconstructions).reshape((-1, VOCAB_SIZE))


def normalize(x, axis):
    """ Normalize a 2D array
    """
    if axis == 1:
        return x / np.linalg.norm(x, axis=1, ord=1)[:, np.newaxis]
    elif axis == 0:
        return x / np.linalg.norm(x, axis=0, ord=1)
    else:
        raise ValueError('Only supports 2D normalization')


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