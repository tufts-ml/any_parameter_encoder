import os
import argparse
import numpy as np
import itertools
import torch
from torch.utils import data
import pyro
from pyro.optim import ExponentialLR, StepLR
from pyro.infer import Trace_ELBO
from pyro.infer.mcmc import NUTS

from dataset import ToyBarsDataset, ToyBarsDocsDataset
from model import APE, APE_VAE
from train import train, train_from_scratch
from evaluate import TimedSVI, TimedMCMC, TimedAVI
from evaluate import get_posterior_predictive_density

import wandb

torch.manual_seed(0)
pyro.set_rng_seed(0)
np.random.seed(0)

from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

parser = argparse.ArgumentParser(description='Results summary')
parser.add_argument('--results_dir', type=str, help='directory of results')
parser.add_argument('--architecture', type=str, help='encoder architecture')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
print(use_cuda)

model_config = {
    'n_hidden_units': 100,
    'n_hidden_layers': 2,
    'results_dir': args.results_dir,
    'alpha': .1,
    'vocab_size': 100,
    'n_topics': 20,
    'use_cuda': use_cuda,
    'architecture': args.architecture,
    # 'scale_type': 'mean',
    'scale_type': 'sample',
    'skip_connections': False,
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
    'epochs': 1,
    'use_cuda': use_cuda,
    'results_dir': args.results_dir,
}

eval_config = {
    'documents': 'data/toy_bar_docs.npy',
    'topics': 'data/test_topics.npy'
}

if __name__ == "__main__":
    wandb.init(sync_tensorboard=True, project="any_parameter_encoder", entity="lily", name=args.results_dir)
    names = []
    inferences = []

    training_set = ToyBarsDocsDataset(training=True, doc_file='data/toy_bar_docs_large.npy', **data_config)
    validation_set = ToyBarsDocsDataset(training=False, doc_file='data/toy_bar_docs_large.npy', **data_config)
    training_generator = data.DataLoader(training_set, **loader_config)
    validation_generator = data.DataLoader(validation_set, **loader_config)
    pyro_scheduler = ExponentialLR({'optimizer': torch.optim.Adam, 'optim_args': {"lr": .01}, 'gamma': 0.95})
    
    ape_training_set = ToyBarsDataset(training=True, doc_file='data/toy_bar_docs.npy', topics_file='data/train_topics.npy', num_models=50000, **data_config)
    ape_validation_set = ToyBarsDataset(training=False, doc_file='data/toy_bar_docs.npy', topics_file='data/valid_topics.npy', num_models=500, **data_config)
    ape_training_generator = data.DataLoader(ape_training_set, **loader_config)
    ape_validation_generator = data.DataLoader(ape_validation_set, **loader_config)
    ape_pyro_scheduler = ExponentialLR({'optimizer': torch.optim.Adam, 'optim_args': {"lr": .01}, 'gamma': 0.95})

    # train APE_VAE from scratch
    ape_vae = APE_VAE(**model_config)
    ape_vae_avi = TimedAVI(ape_vae.model, ape_vae.encoder_guide, pyro_scheduler, loss=Trace_ELBO(), num_samples=100, encoder=ape_vae.encoder)
    ape_vae_avi = train_from_scratch(ape_vae_avi, training_generator, validation_generator, name='ape_vae', **train_config)
    
    # train APE
    ape = APE(**model_config)
    ape_avi = TimedAVI(ape.model, ape.encoder_guide, ape_pyro_scheduler, loss=Trace_ELBO(), num_samples=100, encoder=ape.encoder)
    ape_avi = train(ape_avi, ape_training_generator, ape_validation_generator, name='ape', **train_config)

    pretrained_dict = ape.state_dict()
    
    # train APE_VAE using learned weights from APE
    ape_vae_init = APE_VAE(**model_config)
    model_dict = ape_vae_init.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    ape_vae_init.load_state_dict(pretrained_dict)

    ape_vae_init_avi = TimedAVI(ape_vae_init.model, ape_vae_init.encoder_guide, pyro_scheduler, loss=Trace_ELBO(), num_samples=100, encoder=ape_vae_init.encoder)
    ape_vae_init_avi = train_from_scratch(ape_vae_init_avi, training_generator, validation_generator, name='ape_vae_init', **train_config)
