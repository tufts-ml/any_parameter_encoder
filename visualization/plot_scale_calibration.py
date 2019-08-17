import matplotlib.pyplot as plt
import numpy as np

for data in ['train', 'valid', 'test']:

    vae_scale = np.load('experiments/vae_experiments/10x10/{}_vae_z_scale.npy'.format(data))
    svi_scale = np.load('experiments/vae_experiments/10x10/{}_svi_z_scale.npy'.format(data))
    vae_scale_small = np.load('experiments/vae_experiments/10x10_small2/{}_vae_z_scale.npy'.format(data))
    svi_scale_small = np.load('experiments/vae_experiments/10x10_small2/{}_svi_z_scale.npy'.format(data))

    fig, axes = plt.subplots(2, 2, sharex=True, tight_layout=True)
    axes[0][0].hist(vae_scale.flatten())
    axes[0][1].hist(svi_scale.flatten())
    axes[1][0].hist(vae_scale_small.flatten())
    axes[1][1].hist(svi_scale_small.flatten())
    axes[0][0].set_title('VAE 45-60 words per doc')
    axes[0][1].set_title('SVI 45-60 words per doc')
    axes[1][0].set_title('VAE 5-10 words per doc')
    axes[1][1].set_title('SVI 5-10 words per doc')

    plt.savefig('{}_vae_vs_svi_scale_calibration.pdf'.format(data))