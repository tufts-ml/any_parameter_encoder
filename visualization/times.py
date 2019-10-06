import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_times(results_dir):
    svi_times = pd.read_csv(os.path.join(results_dir, 'svi_loss_curve.csv'), header=None)
    svi_times.columns = ['data', 'step', 'loss', 'time']
    vae_times = pd.read_csv(os.path.join(results_dir, 'vae_times.csv'), header=None)
    vae_times.columns = ['data', 'inference', 'loss', 'time']
    svi_test_times = svi_times[svi_times.data == 'test']
    vae_test_times = vae_times[(vae_times.data == 'test') & (vae_times.inference == 'vae')]
    plt.plot(svi_test_times.time, svi_test_times.loss)
    print(vae_test_times.loss.values[0])
    print(vae_test_times.time.values[0])
    plt.axhline(vae_test_times.loss.values[0], linestyle='--', color="grey", alpha=.5)
    plt.axvline(vae_test_times.time.values[0], linestyle='--', color="grey", alpha=.5)
    plt.plot(vae_test_times.time, vae_test_times.loss, marker='x', markersize=10, color="black")
    plt.savefig(os.path.join(results_dir, 'svi_vs_vae_times.pdf'))

if __name__ == "__main__":
    plot_times('experiments/test_save_restore')