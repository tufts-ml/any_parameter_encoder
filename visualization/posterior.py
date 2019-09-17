import os
import numpy as np
import matplotlib.pyplot as plt

from utils import softmax


def plot_posterior_dense(results_dir, sample_indices, data_name, inference_names, seed=0):
    """ `sample_idx` is a list
    """
    np.random.seed(seed)
    n_cols = 5
    fig, axes = plt.subplots(len(sample_indices) / n_cols, n_cols, sharex=False, sharey=False, tight_layout=True, figsize=(20, 12))
    for i, sample_idx in enumerate(sample_indices):
        for inference, color in zip(inference_names, ['red', 'green', 'purple']):
            if inference == "mcmc":
                samples_unnormalized = np.load(os.path.join(results_dir, '{}_{}_samples.npy'.format(data_name, inference)))[:, sample_idx]
                samples = softmax(samples_unnormalized)
            else:
                z_loc = np.load(os.path.join(results_dir, '{}_{}_z_loc.npy'.format(data_name, inference)))[sample_idx]
                z_scale = np.load(os.path.join(results_dir, '{}_{}_z_scale.npy'.format(data_name, inference)))[sample_idx]
                samples = softmax(np.random.multivariate_normal(z_loc, np.diag(z_scale), size=150))
            means = np.mean(samples, axis=0)
            top_2_idx = np.argpartition(means, -2)[-2:]
            bottom_2_idx = np.argpartition(means, 2)[:2]
            axes[i / n_cols][i % n_cols].scatter(samples[:, top_2_idx[-1]], samples[:, top_2_idx[0]], label=inference, alpha=.2, color=color)
    axes[0][0].legend()
    plt.savefig(os.path.join(results_dir, 'posteriors_{}.png'.format(data_name)))


def plot_posterior(results_dir, sample_idx, data_names, inference_names, scale=1, seed=0):
    np.random.seed(seed)
    fig, axes = plt.subplots(3, len(data_names), sharex=False, sharey=False, tight_layout=True, figsize=(len(data_names) * 4, 12))
    top_2_idx = None
    bottom_2_idx = None
    for i, data in enumerate(data_names):
        for inference, color in zip(inference_names, ['red', 'green', 'purple']):
            if inference == "mcmc":
                samples_unnormalized = np.load(os.path.join(results_dir, '{}_{}_samples.npy'.format(data, inference)))[:, sample_idx]
                samples = softmax(scale * samples_unnormalized)
            else:
                z_loc = np.load(os.path.join(results_dir, '{}_{}_z_loc.npy'.format(data, inference)))[sample_idx]
                z_scale = np.load(os.path.join(results_dir, '{}_{}_z_scale.npy'.format(data, inference)))[sample_idx]
                samples = softmax(scale * np.random.multivariate_normal(z_loc, np.diag(z_scale), size=150))
            means = np.mean(samples, axis=0)
            if top_2_idx is None:
                top_2_idx = np.argpartition(means, -2)[-2:]
            else:
                alt_top_2_idx = np.argpartition(means, -2)[-2:]
                if not np.array_equal(alt_top_2_idx, top_2_idx):
                    if data == 'train':
                        print(alt_top_2_idx)
                        print(top_2_idx)
            if bottom_2_idx is None:
                bottom_2_idx = np.argpartition(means, 2)[:2]
            else:
                alt_bottom_2_idx = np.argpartition(means, -2)[-2:]
                # if not np.array_equal(alt_bottom_2_idx, bottom_2_idx):
                #     print(alt_bottom_2_idx)
                #     print(bottom_2_idx)

            # top_2_idx = np.argpartition(means, -2)[-2:]
            # bottom_2_idx = np.argpartition(means, 2)[:2]
            axes[0][i].scatter(samples[:, top_2_idx[-1]], samples[:, top_2_idx[0]], label=inference, alpha=.2, color=color)
            axes[0][i].set_title("Top two " + data)
            axes[0][i].set_ylim(0, 1)
            axes[0][i].set_xlim(0, 1)
            if inference == 'vae':
                bottom_ax = axes[1][i]
            elif inference in ['svi', 'mcmc']:
                bottom_ax = axes[2][i]
            smallest_weights = samples[:, bottom_2_idx[0]]
            second_smallest_weights = samples[:, bottom_2_idx[1]]
            bottom_ax.scatter(smallest_weights, second_smallest_weights, label=inference, alpha=.2, color=color)
            bottom_ax.set_title("Bottom two " + data)
            # bottom_ax.set_xlim(min(0, smallest_weights.min() * .9), smallest_weights.max() * 1.1)
            # bottom_ax.set_ylim(min(0, smallest_weights.max() * .9), second_smallest_weights.max() * 1.1)
    axes[0][0].legend()
    plt.savefig(os.path.join(results_dir, 'posteriors_{}.png'.format(sample_idx)))


def plot_posterior_v2(results_dir, sample_indices, data_names, inference_names, scale=1, seed=0):
    """ Rows are data examples. Columns are different slices of the posterior. """
    np.random.seed(seed)
    fig, axes = plt.subplots(3, len(data_names), sharex=False, sharey=False, tight_layout=True, figsize=(len(data_names) * 4, 12))
    for i, data in enumerate(data_names):
        fig, axes = plt.subplots(len(sample_indices), len(inference_names),
                                 sharex=False, sharey=False, tight_layout=True, figsize=(len(inference_names) * 4, len(sample_indices) * 4))
        for row, sample_idx in enumerate(sample_indices):
            samples_by_inference = {}
            top_2_idx_by_inference = {}
            for inference in inference_names:
                if inference == "mcmc":
                    samples_unnormalized = np.load(os.path.join(results_dir, '{}_{}_samples.npy'.format(data, inference)))[:, sample_idx]
                    samples = softmax(scale * samples_unnormalized)
                else:
                    z_loc = np.load(os.path.join(results_dir, '{}_{}_z_loc.npy'.format(data, inference)))[sample_idx]
                    z_scale = np.load(os.path.join(results_dir, '{}_{}_z_scale.npy'.format(data, inference)))[sample_idx]
                    samples = softmax(scale * np.random.multivariate_normal(z_loc, np.diag(z_scale), size=150))
                samples_by_inference[inference] = samples
                means = np.mean(samples, axis=0)
                top_2_idx = np.argpartition(means, -2)[-2:]
                top_2_idx_by_inference[inference] = top_2_idx
            for col, inference_name in enumerate(inference_names):
                top_2_idx = top_2_idx_by_inference[inference_name]
                axes[row][col].set_title("Top two topics for " + inference_name)
                axes[row][col].set_ylim(0, 1)
                axes[row][col].set_xlim(0, 1)
                for inference, color in zip(inference_names, ['red', 'green', 'purple']):
                    samples = samples_by_inference[inference]
                    axes[row][col].scatter(samples[:, top_2_idx[-1]], samples[:, top_2_idx[0]], label=inference, alpha=.2, color=color)
        axes[0][0].legend()
        plt.savefig(os.path.join(results_dir, 'posteriors_{}.png'.format(data)))


if __name__ == "__main__":
    # for i in range(10):
    #     plot_posterior('experiments/vae_experiments/naive_scale1/', i,
    #                    ['train', 'valid', 'test'], ['vae', 'svi', 'mcmc'])
    plot_posterior_v2('experiments/naive_mean4/', [3, 4, 5], ['train', 'valid', 'test'], ['vae', 'svi', 'mcmc'])