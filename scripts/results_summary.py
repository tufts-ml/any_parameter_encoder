import os
import csv
import pandas as pd
import argparse
import numpy as np


def normalize_posterior_predictive(df, num_words_per_dataset, dataset_names):
    subsets = []
    for name in dataset_names:
        # num_words = data.sum()
        # num_words = sum([doc.sum() for doc in data])
        num_words = num_words_per_dataset[name]
        print(num_words)
        subset = df[df.dataset==name]
        subset['posterior_predictive_density'] /= num_words
        subsets.append(subset)
    return pd.concat(subsets)


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
        try:
            num_words_per_dataset = {}
            with open(os.path.join(args.results_dir, 'num_words.csv'), 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    name, num_words = row
                    num_words_per_dataset[name] = int(num_words)
            print('num_words_per_dataset', num_words_per_dataset)
        except:
            print('Could not find a num_words.csv. Assuming num words.')
            datasets = [
                documents[0] * 200 + documents[1] * 100,
                documents[:60] * 5,
                documents[:60] * 5]
            num_words_per_dataset = {name: docs.sum() for name, docs in zip(dataset_names, datasets)}
        df = normalize_posterior_predictive(df, num_words_per_dataset, dataset_names)
        df.to_csv(os.path.join(args.results_dir, 'results_normalized.csv'), header=False, index=False)
    print(df.groupby(['dataset', 'inference']).posterior_predictive_density.agg(
        {'std':'std', 'mean':'mean', 'max': 'max'}))