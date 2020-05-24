import os
import argparse
import numpy as np
import pandas as pd
import itertools
from copy import deepcopy
import torch
from torch.utils import data
import pyro
from pyro.optim import ExponentialLR, StepLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
from pyro.infer import Trace_ELBO, TraceMeanField_ELBO
from pyro.infer.mcmc import NUTS

from dataset import ToyBarsDataset, ToyBarsDocsDataset, NonToyBarsDataset
from model import APE, APE_VAE
from train import train, train_from_scratch, get_val_loss
from evaluate import TimedSVI, TimedMCMC, TimedAVI
from evaluate import get_posterior_predictive_density

import wandb

torch.manual_seed(0)
np.random.seed(0)

from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

parser = argparse.ArgumentParser(description='Results summary')
parser.add_argument('--results_dir', type=str, help='directory of results')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
print(use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")

model_config = {
    'n_hidden_units': 100,
    'n_hidden_layers': 2,
    'results_dir': args.results_dir,
    'alpha': .1,
    'vocab_size': 100,
    'n_topics': 20,
    'use_cuda': use_cuda,
    'architecture': None,
    'scale_type': 'sample',
    'skip_connections': False,
    'model_type': 'nvdm'
}

data_config = {
    'n_topics': 20,
    'vocab_size': 100,
    'alpha': .1,
    'use_cuda': use_cuda,
    'generate': False
}

loader_config = {
    'batch_size': 500,
    'shuffle': True,
    'num_workers': 0}

train_config = {
    'epochs': 100,
    'use_cuda': use_cuda,
    'results_dir': args.results_dir,
    'scaled': True
}

eval_config = {
    'documents': 'data/toy_bar_docs.npy',
    'topics': 'data/test_topics.npy'
}

if __name__ == "__main__":
    names = []
    inferences = []

    true_ape_validation_set = NonToyBarsDataset(training=True, doc_file='data/non_toy_bars_docs.npy', topics_file='data/non_toy_bars_topics.npy', num_models=1, num_docs=5000, **data_config)
    true_ape_validation_generator = data.DataLoader(true_ape_validation_set, **loader_config)
    true_ape_validation_set_many_words = NonToyBarsDataset(training=True, doc_file='data/non_toy_bars_docs_many_words.npy', topics_file='data/non_toy_bars_topics.npy', num_models=1, num_docs=5000, avg_num_words=500, **data_config)
    true_ape_validation_generator_many_words = data.DataLoader(true_ape_validation_set_many_words, **loader_config)

    losses_to_record = {}

    models = ['avitm', 'nvdm']
    # test APE, no training
    ape_model_config = deepcopy(model_config)
    values = []
    for seed in range(10):
        pyro.set_rng_seed(seed)
        for combo in itertools.product(['true_topics'], models):
            for loss in [Trace_ELBO, TraceMeanField_ELBO]:
                for data_size in ['small', 'large']:
                    ape_pyro_scheduler = StepLR({'optimizer': torch.optim.Adam, 'optim_args': {"lr": .01}, "step_size": 250, "gamma": .5})
                    topic_type, model_type, architecture = combo
                    ape_model_config['model_type'] = model_type
                    ape_model_config['architecture'] = architecture
                    ape_model_config['n_hidden_layers'] = 0
                    ape_model_config['n_hidden_units'] = ape_model_config['n_topics']

                    if topic_type == 'true_topics':
                        if data_size == 'large':
                            val_gen = true_ape_validation_generator_many_words
                        else:
                            val_gen = true_ape_validation_generator

                    ape_vae = APE(**ape_model_config)

                    pyro_scheduler = StepLR({'optimizer': torch.optim.Adam, 'optim_args': {"lr": .05}, 'step_size': 200, 'gamma': 0.95})
                    svi = TimedSVI(ape_vae.model, ape_vae.mean_field_guide, pyro_scheduler, loss=loss(), num_samples=100) #, num_steps=100000)
                    n_epochs = 10000
                    svi = train(svi, val_gen, val_gen, pyro_scheduler, **{'epochs': n_epochs, 'use_cuda': use_cuda, 'results_dir': args.results_dir})
                    val_loss = get_val_loss(svi, val_gen, use_cuda, device, scaled=True)
                    print(combo, val_loss)
                    values.append([topic_type, model_type, 'svi_mean_field', loss.__name__, data_size, val_loss, seed])
    df = pd.DataFrame(values)
    df.columns = ['topic_type', 'model_type', 'architecture', 'metric', 'data_size', 'loss', 'seed']
    df.to_csv('no_training_non_toy_bars_svi.csv')