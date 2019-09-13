import os
import pandas as pd
import argparse
import numpy as np
from scripts.normalize_posterior_predictive import normalize_posterior_predictive


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Results summary')
    parser.add_argument('results_dir', type=str, help='Experiment Directory')
    parser.add_argument('--normalized', help='Experiment Directory', action='store_true')
    args = parser.parse_args()

    print(args.results_dir)
    df = pd.read_csv(os.path.join(args.results_dir, 'results.csv'), header=None)
    df.columns = ['inference', 'model', 'dataset', 'n_hidden_layers', 'n_hidden_units', 'posterior_predictive_density']
    if args.normalized:
        dataset_names = ['train', 'valid', 'test']
        documents_file = os.path.join(args.results_dir, 'documents.npy')
        if not os.path.exists(documents_file):
            documents_file = os.path.join('experiments', 'documents.npy')
        documents = np.load(documents_file)
        # valid = np.load(os.path.join(results_dir, 'valid_documents.npy'))
        datasets = [documents * 200, documents * 5, documents * 5]
        df = normalize_posterior_predictive(df, datasets, dataset_names)
        df.to_csv(os.path.join(args.results_dir, 'results_normalized.csv'), header=False, index=False)
    print(df.groupby(['dataset', 'inference']).posterior_predictive_density.agg(
        {'std':'std', 'mean':'mean', 'max': 'max'}))