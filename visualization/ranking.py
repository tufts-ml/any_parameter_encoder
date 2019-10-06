import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_svi_vs_vae_elbo(results_dir):
    df = pd.read_csv(os.path.join(results_dir, 'elbos.csv'))
    g = sns.FacetGrid(df, col="Dataset")
    g.map(plt.scatter, "Decoder-aware encoder ELBO", "SVI ELBO").add_legend()
    plt.savefig(os.path.join(results_dir, 'elbo_ranking.png'))
    plt.close()
    g = sns.FacetGrid(df, col="Dataset")
    g.map(plt.scatter, "Standard encoder ELBO", "SVI ELBO").add_legend()
    plt.savefig(os.path.join(results_dir, 'elbo_ranking_single.png'))


def plot_svi_vs_vae_elbo_v1(results_dir, posterior_predictive=False):
    if posterior_predictive:
        df = pd.read_csv(os.path.join(results_dir, 'posterior_predictives.csv'))
    else:
        df = pd.read_csv(os.path.join(results_dir, 'elbos.csv'))
    train_test = df[df['Dataset'].isin(['train', 'test'])]
    # min_elbo = min(train_test['Standard encoder ELBO'].min(), train_test['Decoder-aware encoder ELBO'].min(), train_test['SVI ELBO'].min())
    # print(train_test['Standard encoder ELBO'].min(), train_test['Decoder-aware encoder ELBO'].min(), train_test['SVI ELBO'].min())
    # max_elbo = max(train_test['Standard encoder ELBO'].max(), train_test['Decoder-aware encoder ELBO'].max(), train_test['SVI ELBO'].max())
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 4), tight_layout=True)

    train = df[df['Dataset']=='train']
    axes[0].scatter(x=train['Standard encoder ELBO'], y=train['SVI ELBO'], color='red', label='Standard encoder')
    axes[0].scatter(x=train['Decoder-aware encoder ELBO'], y=train['SVI ELBO'], color='blue', label='Decoder-aware encoder')
    x = np.linspace(train_test['SVI ELBO'].min(), train_test['SVI ELBO'].max(), 100)
    axes[0].plot(x, x, '--')
    axes[0].set_title('Train')

    test = df[df['Dataset']=='test']
    axes[1].scatter(x=test['Standard encoder ELBO'], y=test['SVI ELBO'], color='red', label='Standard encoder')
    axes[1].scatter(x=test['Decoder-aware encoder ELBO'], y=test['SVI ELBO'], color='blue', label='Decoder-aware encoder')
    axes[1].set_title('Test')
    axes[1].plot(x, x, '--')
    # plt.axis([min_elbo, max_elbo, min_elbo, max_elbo])
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    if posterior_predictive:
        xlabel = 'Posterior predictive log-likelihood (encoder)'
        ylabel = 'Posterior predictive log-likelihood (SVI)'
    else:
        xlabel = 'ELBO (encoder)'
        ylabel = 'ELBO (SVI)'
    # axes[0].set_xlabel(xlabel)
    # axes[1].set_xlabel(xlabel)
    # plt.xlabel(xlabel)
    fig.text(0.55, -.005, xlabel, ha='center')
    axes[0].set_ylabel(ylabel)
    axes[1].set_ylabel(ylabel)
    plt.legend()
    if posterior_predictive:
        final_filename = 'posterior_predictive_ranking.pdf'
    else:
        final_filename = 'elbo_ranking.pdf'
    plt.savefig(os.path.join(results_dir, final_filename), bbox_inches="tight")


if __name__ == "__main__":
    plot_svi_vs_vae_elbo_v1('experiments/final/r6_test1')