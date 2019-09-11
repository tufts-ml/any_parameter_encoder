import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Results summary')
    parser.add_argument('results_dir', type=str, help='Experiment Directory')
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.results_dir, 'elbo_vs_m.csv'))
    df.dropna(inplace=True, axis=0)
    df.elbo = df.elbo.astype(float)
    df.m = df.m.astype(float)
    g = sns.FacetGrid(df, col='data')
    g.map(sns.lineplot,'m', 'elbo', 'inference')
    plt.legend()
    plt.savefig(os.path.join(args.results_dir, 'elbo_vs_m.pdf'))