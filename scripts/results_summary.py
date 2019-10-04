import os
import pandas as pd
import argparse
import numpy as np


def normalize_posterior_predictive(df, datasets, dataset_names):
    subsets = []
    for name, data in zip(dataset_names, datasets):
        # num_words = data.sum()
        num_words = sum([doc.sum() for doc in data])
        print(num_words)
        subset = df[df.dataset==name]
        subset['posterior_predictive_density'] /= num_words
        subsets.append(subset)
    return pd.concat(subsets)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Results summary')
    parser.add_argument('results_dir', type=str, help='Experiment Directory')
    parser.add_argument('--normalized', help='Experiment Directory', action='store_true')
    parser.add_argument('--full_data', help='Experiment Directory', action='store_true')
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
        if args.full_data:
            datasets = [documents * 200, documents * 5, documents * 5]
        else:
            datasets = [
                documents[0] * 200 + documents[1] * 100,
                documents[:60] * 5,
                documents[:60] * 5]
        df = normalize_posterior_predictive(df, datasets, dataset_names)
        df.to_csv(os.path.join(args.results_dir, 'results_normalized.csv'), header=False, index=False)
    print(df.groupby(['dataset', 'inference']).posterior_predictive_density.agg(
        {'std':'std', 'mean':'mean', 'max': 'max'}))