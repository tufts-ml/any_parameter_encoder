import os
import pandas as pd
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Results summary')
    parser.add_argument('results_dir', type=str, help='Experiment Directory')
    args = parser.parse_args()

    results_dir = 'naive_reconstructions'
    print(args.results_dir)
    df = pd.read_csv(os.path.join(args.results_dir, 'results.csv'))
    df.columns = ['inference', 'model', 'dataset', 'n_hidden_layers', 'n_hidden_units', 'posterior_predictive_density']
    print(df.groupby(['dataset', 'inference']).posterior_predictive_density.agg(
        {'std':'std', 'mean':'mean', 'max': 'max'}))