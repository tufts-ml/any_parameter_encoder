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

from dataset import ToyBarsDataset, ToyBarsDocsDataset
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
parser.add_argument('--architecture', type=str)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--model_type', type=str)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
print(use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")

model_config = {
    'n_hidden_units': 100,
    'n_hidden_layers': 2,
    'results_dir': '',
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
    'avg_num_words':5000,
    'exact_toy_bars': True
}

loader_config = {
    'batch_size': 500,
    'shuffle': True,
    'num_workers': 0}

train_config = {
    'epochs': 50,
    'use_cuda': use_cuda,
    'results_dir': '',
    'scaled': True
}

eval_config = {
    'documents': 'data/toy_bar_docs.npy',
    'topics': 'data/test_topics.npy'
}

if __name__ == "__main__":
    project_name = "ape_vae_sweep"
    combo = [args.model_type, args.architecture, str(args.learning_rate)]
    name = 'ape_vae_' + "_".join(combo)
    default_config = {
        'model_type': 'avitm',
        'architecture': 'standard',
        'learning_rate': .1
    }
    wandb.init(sync_tensorboard=True, project=project_name, entity="lily", name=name, config=default_config)
    print('Starting APE_VAE')
    training_set = ToyBarsDocsDataset(doc_file='data/toy_bar_docs_big_train.npy', num_docs=50000, **data_config)
    validation_set = ToyBarsDocsDataset(doc_file='data/toy_bar_docs_big_val.npy', num_docs=5000, **data_config)
    training_generator = data.DataLoader(training_set, **loader_config)
    validation_generator = data.DataLoader(validation_set, **loader_config)

    # test APE_VAE with training
    ape_vae_model_config = deepcopy(model_config)
    # ape_vae_pyro_scheduler = CosineAnnealingWarmRestarts({'optimizer': torch.optim.Adam, 'T_0': 5000, 'optim_args': {"lr": .005}})
    ape_vae_pyro_scheduler = StepLR(
        {'optimizer': torch.optim.Adam,
         'optim_args': {"lr": args.learning_rate},
         "step_size": 250, "gamma": .5
        })
    ape_vae_model_config['model_type'] = args.model_type
    ape_vae_model_config['architecture'] = args.architecture
    

    ape_vae = APE_VAE(**ape_vae_model_config)
    ape_vae_avi = TimedAVI(ape_vae.model, ape_vae.encoder_guide, ape_vae_pyro_scheduler, loss=Trace_ELBO(retain_graph=True), num_samples=100, encoder=ape_vae.encoder)
    ape_vae_avi = train_from_scratch(ape_vae_avi, training_generator, validation_generator, ape_vae_pyro_scheduler, name='', **train_config)
    torch.save(ape_vae.state_dict(), os.path.join('_'.join(combo), 'ape_vae.dict'))
    print('ape_vae finished')

    train_loss = get_val_loss(ape_vae_avi, training_generator, use_cuda, device, scaled=True)
    val_loss = get_val_loss(ape_vae_avi, validation_generator, use_cuda, device, scaled=True)

    metrics = {'train_loss': train_loss, 'val_loss': val_loss}
    wandb.log(metrics)