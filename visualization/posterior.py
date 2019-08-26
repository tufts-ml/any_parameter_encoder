import os
import numpy as np
import matplotlib.pyplot as plt

from utils import softmax


def plot_posterior(results_dir, sample_idx, data_names, inference_names):
    fig, axes = plt.subplots(3, len(data_names), sharex=False, sharey=False, tight_layout=True, figsize=(len(data_names) * 4, 12))
    for i, data in enumerate(data_names):
        for inference, color in zip(inference_names, ['red', 'green', 'purple']):
            if inference == "mcmc":
                samples_unnormalized = np.load(os.path.join(results_dir, '{}_{}_samples.npy'.format(data, inference)))[:, sample_idx]
                samples = softmax(samples_unnormalized)
            else:
                z_loc = np.load(os.path.join(results_dir, '{}_{}_z_loc.npy'.format(data, inference)))[sample_idx]
                z_scale = np.load(os.path.join(results_dir, '{}_{}_z_scale.npy'.format(data, inference)))[sample_idx]
                samples = softmax(np.random.multivariate_normal(z_loc, np.diag(z_scale), size=150))
            means = np.mean(samples, axis=0)
            top_2_idx = np.argpartition(means, -2)[-2:]
            bottom_2_idx = np.argpartition(means, 2)[:2]
            axes[0][i].scatter(samples[:, top_2_idx[-1]], samples[:, top_2_idx[0]], label=inference, alpha=.2, color=color)
            axes[0][i].set_title("Top two " + data)
            if inference == 'vae':
                bottom_ax = axes[1][i]
            elif inference in ['svi', 'mcmc']:
                bottom_ax = axes[2][i]
            smallest_weights = samples[:, bottom_2_idx[0]]
            second_smallest_weights = samples[:, bottom_2_idx[1]]
            bottom_ax.scatter(smallest_weights, second_smallest_weights, label=inference, alpha=.2, color=color)
            bottom_ax.set_title("Bottom two " + data)
            bottom_ax.set_xlim(min(0, smallest_weights.min() * .9), smallest_weights.max() * 1.1)
            bottom_ax.set_ylim(min(0, smallest_weights.max() * .9), second_smallest_weights.max() * 1.1)
    axes[0][0].legend()
    plt.savefig(os.path.join(results_dir, 'posteriors_{}.png'.format(sample_idx)))


if __name__ == "__main__":
    for i in [99, 100]:
        plot_posterior('experiments/vae_experiments/debugging/', i,
                       ['train', 'valid'], ['mcmc'])