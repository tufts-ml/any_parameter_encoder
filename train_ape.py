import os
import argparse
import numpy as np
import itertools
from copy import deepcopy
import torch
from torch.utils import data
import pyro
from pyro.optim import ExponentialLR, StepLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
from pyro.infer import Trace_ELBO
from pyro.infer.mcmc import NUTS

from ape_dataset import APEDataset
from model import APE, APE_VAE
from train import train, train_from_scratch, train_ape
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
parser.add_argument('--architecture', type=str, help='encoder architecture', default='template')
parser.add_argument('--test', dest='test', action='store_true')
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
    'scale_type': 'sample',
    'skip_connections': False,
    'model_type': 'avitm'
}

data_config = {
    'n_topics': 20,  # number of topics in a given model
    'vocab_size': 100,
    'use_cuda': use_cuda,
    'num_models': 300,  # number of topic sets, i.e. number of models
    'num_docs': 150
}

if args.test:
    data_config['num_models'] = 3
    data_config['num_docs'] = 2

loader_config = {
    # 'batch_size': 500,
    'batch_size': 2,
    'shuffle': True,
    'num_workers': 0}

train_config = {
    'epochs': 100,
    'use_cuda': use_cuda,
    'results_dir': args.results_dir,
}

eval_config = {
    'documents': 'data/toy_bar_docs.npy',
    'topics': 'data/test_topics.npy'
}

if __name__ == "__main__":
    wandb.init(sync_tensorboard=True, project="any_parameter_encoder", entity="lily", name=args.results_dir)
    
    if not os.path.exists('ape_data'):
        os.mkdir('ape_data')

    train_data_config = deepcopy(data_config)
    if args.test:
        train_data_config.update({'num_models': 4, 'num_docs': 3})
    else:
        train_data_config.update({'num_models': 10000, 'num_docs': 5000})
    ape_training_set = APEDataset(doc_file='ape_data/same_docs.npy', topics_file='ape_data/same_topics.npy', **train_data_config)
    data_generators = {'train': data.DataLoader(ape_training_set, **loader_config)}
    for combo in itertools.product(['same', 'sim', 'diff'], ['same', 'sim', 'diff']):
        doc, topic = combo
        dataset = APEDataset(doc_file=f'ape_data/{doc}_docs.npy', topics_file=f'ape_data/{topic}_topics.npy', **data_config)
        data_generators[combo] = data.DataLoader(dataset, **loader_config)

    # train APE
    ape = APE(**model_config)
    ape_pyro_scheduler = CosineAnnealingWarmRestarts({'optimizer': torch.optim.Adam, 'T_0': 500, 'optim_args': {"lr": .005}})
    ape_avi = TimedAVI(ape.model, ape.encoder_guide, ape_pyro_scheduler, loss=Trace_ELBO(), num_samples=100, encoder=ape.encoder)
    ape_train_config = deepcopy(train_config)
    ape_train_config['epochs'] = 1
    ape_avi = train_ape(ape_avi, data_generators, ape_pyro_scheduler, name='ape', **ape_train_config)
    torch.save(ape.state_dict(), os.path.join(args.results_dir, 'ape.dict'))
    print('ape finished')