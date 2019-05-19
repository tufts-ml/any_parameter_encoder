import os
import numpy as np
import torch
import pandas as pd
import csv

from pyro.optim import StepLR
from pyro.infer import SVI, Trace_ELBO, TracePredictive
from common import evaluate_log_predictive_density

from datasets.load import load_toy_bars
from models.lda_lognorm import VAE_pyro
from common import train_save_VAE, save_loglik_to_csv, save_reconstruction_array
from visualization.reconstructions import plot_side_by_side_docs
from utils import softmax

# where to write the results
results_dir = 'mdreviews1'
results_file = 'reconstruction_by_sparsity_scale.csv'

# global params
n_topics = 20
num_examples = 10
sample_idx = list(range(10))

# Amazon product reviews data
dataset_names = ['train', 'valid', 'test']
train = np.load('datasets/mdreviews/train.npy')
valid = np.load('datasets/mdreviews/valid.npy')
test = np.load('datasets/mdreviews/test.npy')
datasets = [train, valid, test]
data_tr = datasets[0]

# various VAEs, all with fixed decoders
model_config = {
    'topic_init': 'resources/mdreviews_topics.npy',
    # 'topic_init': 'resources/topics_10x10.npy',
    # 'topic_init': None,
    'topic_trainable': False,
    'vocab_size': datasets[0].shape[1],
    'n_topics': n_topics,
    'results_dir': results_dir,
    'results_file': results_file,
    'inference': 'vae',
    'model_name': 'lda_scale',
    'enc_topic_init': None,
    'enc_topic_trainable': True,
    'scale_trainable': False,
    'n_hidden_layers': 5,
    'n_hidden_units': 50,
    'n_samples': 10,
    'decay_rate': .5,
    'decay_steps': 1000,
    'starting_learning_rate': .01,
}

df = pd.read_csv(os.path.join(results_dir, 'results_by_sparsity1.csv'))
df.columns = ['num_topics', 'dataset', 'data_idx']
sparsity_dict = dict(df.groupby(['dataset', 'num_topics'])['data_idx'].apply(list))


datasets = datasets[1:2]
dataset_names = dataset_names[1:2]
# load the VAE into pyro for evaluation
vae = VAE_pyro(**model_config)
state_dict = vae.load()
vae.load_state_dict(state_dict)
with open(os.path.join(results_dir, results_file), 'a') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['dataset', 'num_topics', 'posterior_predictive_density'])
    for data_name, data in zip(dataset_names, datasets):
        for i in range(n_topics):
            if (data_name, i) in sparsity_dict.keys():
                print((data_name, i))
                docs = data[sparsity_dict[(data_name, i)]]
                num_words = docs.sum()
                print('num_words', num_words)
                docs = torch.from_numpy(docs.astype(np.float32))
                pyro_scheduler = StepLR({'optimizer': torch.optim.Adam, 'optim_args': {"lr": .1}, 'step_size': 10000, 'gamma': 0.95})
                # pyro scheduler doesn't have any effect in the VAE case since we never take any optimization steps
                vae_svi = SVI(vae.model, vae.encoder_guide, pyro_scheduler, loss=Trace_ELBO(), num_steps=100, num_samples=10)
                posterior = vae_svi.run(docs)
                posterior_predictive = TracePredictive(vae.model, posterior, num_samples=10)
                posterior_predictive_traces = posterior_predictive.run(docs)
                # get the posterior predictive log likelihood
                posterior_predictive_density = evaluate_log_predictive_density(posterior_predictive_traces)
                posterior_predictive_density = float(posterior_predictive_density.detach().numpy())
                posterior_predictive_density /= num_words
                print('posterior_predictive', posterior_predictive_density)
                csv_writer.writerow([data_name, i, posterior_predictive_density])

