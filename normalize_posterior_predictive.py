import os
import pandas as pd
import numpy as np
from datasets.load import load_toy_bars


def normalize_posterior_predictive(df, datasets, dataset_names):
    subsets = []
    for name, data in zip(dataset_names, datasets):
        num_words = data.sum()
        print(num_words)
        subset = df[df.dataset==name]
        subset['posterior_predictive_density'] /= num_words
        subsets.append(subset)
    return pd.concat(subsets)
#
# # Amazon reviews
# results_dir = 'mdreviews'
# results_csv = 'results.csv'
#
# df = pd.read_csv(os.path.join(results_dir, results_csv), header=None)
# df.columns = ['inference', 'model', 'dataset', 'n_hidden_layers', 'n_hidden_units', 'posterior_predictive_density']
# df = df[df.model.isin(['lda_orig', 'lda_scale', 'lda_orig_hallucinations', 'lda_scale_hallucinations'])]
#
# dataset_names = ['train', 'valid', 'test']
# train = np.load('datasets/mdreviews/train.npy')
# valid = np.load('datasets/mdreviews/valid.npy')
# test = np.load('datasets/mdreviews/test.npy')
# datasets = [train, valid, test]
# data_tr = datasets[0]
# datasets[0] = data_tr[:792]
# datasets[2] = datasets[2][:792]
# df = normalize_posterior_predictive(df, datasets, dataset_names)
# df.to_csv(os.path.join(results_dir, 'results_normalized.csv'), header=False, index=False)

# Toy bars
results_dir = 'experiments/vae_experiments/problem_toy_bars5'
results_csv = 'results.csv'

df = pd.read_csv(os.path.join(results_dir, results_csv), header=None)
df.columns = ['inference', 'model', 'dataset', 'n_hidden_layers', 'n_hidden_units', 'posterior_predictive_density']
# df = df[df.model.isin(['lda_orig', 'lda_scale', 'lda_orig_hallucinations', 'lda_scale_hallucinations'])]

# could be automated; specified for now to make sure the plots come in the right order
# dataset_names = ['train', 'valid', 'test', 'test_single', 'test_double', 'test_triple']
# datasets = load_toy_bars('toy_bars_10x10')
# data_tr = datasets[0]
# datasets[0] = data_tr[:1000]
# df = normalize_posterior_predictive(df, datasets, dataset_names)
# df.to_csv(os.path.join(results_dir, 'results_normalized.csv'), header=False, index=False)


datasets = load_toy_bars('toy_bars_10x10')
data_tr = datasets[0]
data_tr_single = data_tr[np.count_nonzero(data_tr, axis=1) <= 10]
data_tr_double = data_tr[np.count_nonzero(data_tr, axis=1) > 10]
datasets = [data_tr_single[:1000], data_tr_double[:1000]] + datasets[1:]
dataset_names = ['train_single', 'train_double', 'valid', 'test', 'test_single', 'test_double', 'test_triple']
df = normalize_posterior_predictive(df, datasets, dataset_names)
df.to_csv(os.path.join(results_dir, 'results_normalized.csv'), header=False, index=False)