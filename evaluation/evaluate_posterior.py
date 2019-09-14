import os
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


def reconstruct_data(posterior, vae, topics):
    """Generate data from traces
    """
    # encode image x
    reconstructions = []
    # each tr contains a trace for every datapoint
    for tr in posterior.exec_traces:
        z_loc = tr.nodes['latent']['value']
        # decode the image (note we don't sample in image space)
        reconstruction = vae.decoder(z_loc, topics).detach().numpy()
        reconstructions.append(reconstruction)
    return np.array(reconstructions)
    # we want each set of reconstructions to be a column
    # reconstructions = zip(*reconstructions)
    # return np.array(reconstructions).reshape((-1, VOCAB_SIZE))


def kl_enc_svi(results_dir, data_names=['train', 'valid', 'test']):
    all_kl_q_p = []
    all_kl_p_q = []
    for data_name in data_names:
        svi_loc = np.load(os.path.join(results_dir, '{}_{}_z_loc.npy'.format(data_name, 'svi')))
        svi_scale = np.load(os.path.join(results_dir, '{}_{}_z_scale.npy'.format(data_name, 'svi')))
        vae_loc = np.load(os.path.join(results_dir, '{}_{}_z_loc.npy'.format(data_name, 'vae')))
        vae_scale = np.load(os.path.join(results_dir, '{}_{}_z_scale.npy'.format(data_name, 'vae')))
        kl_divs_q_p = map(kl_mult_gauss, vae_loc, vae_scale, svi_loc, svi_scale)
        kl_divs_p_q = map(kl_mult_gauss, svi_loc, svi_scale, vae_loc, vae_scale)
        print("KL(q||p)", data_name, np.mean(kl_divs_q_p), np.std(kl_divs_q_p))
        print("KL(p||q)", data_name, np.mean(kl_divs_p_q), np.std(kl_divs_p_q))
        all_kl_q_p.append(kl_divs_q_p)
        all_kl_p_q.append(kl_divs_p_q)
    return all_kl_q_p, all_kl_p_q


def kl_mult_gauss(q_mu, q_sigma, p_mu, p_sigma):
    """Compare two multivariate normal (or log normal) distributions """
    dim = len(q_mu)
    kl = 0.5 * (
            np.sum(np.divide(q_sigma, p_sigma)) +
            np.sum(
                np.multiply(
                    np.divide((q_mu - p_mu), p_sigma),
                    (q_mu - p_mu)
                ),
            )
            - dim
            + np.sum(np.log(p_sigma))
            - np.sum(np.log(q_sigma))
        )
    return kl