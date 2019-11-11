import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_times(results_dir):
    svi_times = pd.read_csv(os.path.join(results_dir, 'svi_loss_curve.csv'), header=None)
    svi_times.columns = ['data', 'step', 'loss', 'time', 'n_runs']
    vae_times = pd.read_csv(os.path.join(results_dir, 'vae_times.csv'), header=None)
    vae_times.columns = ['data', 'inference', 'loss', 'time']
    svi_test_times = svi_times[svi_times.data == 'test']
    # find the time with the best run
    best_run = svi_test_times.loc[svi_test_times.loss.idxmax()].n_runs
    print(best_run)
    svi_test_times = svi_test_times[svi_test_times.n_runs==best_run]
    vae_test_times = vae_times[(vae_times.data == 'test') & (vae_times.inference == 'vae')]
    plt.plot(svi_test_times.time, svi_test_times.loss)
    print(vae_test_times.loss.values[0])
    print(vae_test_times.time.values[0])
    plt.axhline(vae_test_times.loss.values[0], linestyle='--', color="grey", alpha=.5)
    plt.axvline(vae_test_times.time.values[0], linestyle='--', color="grey", alpha=.5)
    plt.plot(vae_test_times.time, vae_test_times.loss, marker='x', markersize=10, color="black")
    plt.savefig(os.path.join(results_dir, 'svi_vs_vae_times.pdf'))


def plot_times_v1(results_dir):
    svi_times = pd.read_csv(os.path.join(results_dir, 'svi_loss_curve.csv'), header=None)
    svi_times.columns = ['data', 'step', 'loss', 'time', 'n_runs']
    vae_times = pd.read_csv(os.path.join(results_dir, 'vae_times.csv'), header=None)
    vae_times.columns = ['data', 'inference', 'loss', 'time']
    vae_test_times = vae_times[(vae_times.data == 'test') & (vae_times.inference == 'vae')]
    for data, label, color in zip(['test', 'test_warmstart'], ['VI', 'VI with encoder init'], ['black', 'green']):
        svi_test_times = svi_times[svi_times.data == data]
        best_run = svi_test_times.loc[svi_test_times.loss.idxmax()].n_runs
        print(data, label, best_run)
        svi_test_times = svi_test_times[svi_test_times.n_runs==best_run]
        print(svi_test_times.loss.max())
        plt.plot(svi_test_times.time * 1000, svi_test_times.loss, label=label, color=color)
        # if style == ':':
        #     plt.plot(svi_test_times.time * 1000, svi_test_times.loss, label=label, linestyle=style, color='black')
        # else:
        #     plt.plot(svi_test_times.time * 1000, svi_test_times.loss, label=label, linestyle=style, color='black')
        if data == 'test':
            idx = svi_test_times[svi_test_times.loss > vae_test_times.loss.values[0]].loss.idxmin()
            svi_time_to_match = svi_test_times.loc[idx].time * 1000
            plt.axvline(svi_time_to_match, linestyle='--', color="grey", alpha=.5)
    print('time', svi_time_to_match)
    # plt.axhline(vae_test_times.loss.values[0], xmin=vae_test_times.time.values[0] / 1000, xmax=svi_time_to_match / 1000, linestyle='--', color="grey", alpha=.5)
    plt.axvline(vae_test_times.time.values[0] * 1000, linestyle='--', color="grey", alpha=.5)
    plt.plot(vae_test_times.time * 1000, vae_test_times.loss, marker='x', markersize=20, color="red")
    plt.xlim([0, 1000])
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('ELBO')
    plt.savefig(os.path.join(results_dir, 'svi_vs_vae_times_zoomed.pdf'), bbox_inches="tight")
    plt.close()

    for data, label in zip(['test', 'test_warmstart'], ['VI', 'VI with encoder init']):
        svi_test_times = svi_times[svi_times.data == data]
        best_run = svi_test_times.loc[svi_test_times.loss.idxmax()].n_runs
        print(best_run)
        svi_test_times = svi_test_times[svi_test_times.n_runs==best_run]
        print(svi_test_times.loss.max())
        if data == 'test_warmstart':
            print(data)
            plt.axhline(svi_test_times.loss.max(), linestyle='--', color="grey", alpha=.5)
        plt.plot(svi_test_times.time, svi_test_times.loss, label=label)
    # plt.axhline(vae_test_times.loss.values[0], linestyle='--', color="grey", alpha=.5)
    plt.plot(vae_test_times.time, vae_test_times.loss, marker='x', markersize=10, color="black")
    plt.xlabel('Time (seconds)')
    plt.ylabel('ELBO')
    plt.savefig(os.path.join(results_dir, 'svi_vs_vae_times.pdf'), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    sns.set_style("ticks") # or 'white' or 'whitegrid'
    sns.set_context("notebook", font_scale=1.8) # magnify fonts by 2x
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times New Roman"
    plot_times_v1('generalize3')