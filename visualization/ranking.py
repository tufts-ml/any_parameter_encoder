import os
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