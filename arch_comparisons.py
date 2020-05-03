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

from dataset import ToyBarsDataset, ToyBarsDocsDataset
from model import APE, APE_VAE
from train import train, train_from_scratch, get_val_loss
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
    wandb.init(sync_tensorboard=True, project="any_parameter_encoder", entity="lily", name=args.results_dir)
    names = []
    inferences = []

    training_set = ToyBarsDocsDataset(training=True, doc_file='data/toy_bar_docs_large.npy', subset_docs=50000, **data_config)
    validation_set = ToyBarsDocsDataset(training=False, doc_file='data/toy_bar_docs_large.npy', subset_docs=50000, **data_config)
    training_generator = data.DataLoader(training_set, **loader_config)
    validation_generator = data.DataLoader(validation_set, **loader_config)
    
    ape_training_set = ToyBarsDataset(training=True, doc_file='data/toy_bar_docs_large.npy', topics_file='data/train_topics.npy', num_models=20000, subset_docs=5000, **data_config)
    ape_validation_set = ToyBarsDataset(training=False, doc_file='data/toy_bar_docs_large.npy', topics_file='data/valid_topics.npy', num_models=50, subset_docs=5000, **data_config)
    ape_training_generator = data.DataLoader(ape_training_set, **loader_config)
    ape_validation_generator = data.DataLoader(ape_validation_set, **loader_config)

    true_ape_training_set = ToyBarsDataset(training=True, doc_file='data/toy_bar_docs_large.npy', topics_file='data/true_topics.npy', num_models=1, subset_docs=5000, **data_config)
    true_ape_training_generator = data.DataLoader(true_ape_training_set, **loader_config)
    true_ape_validation_set = ToyBarsDataset(training=False, doc_file='data/toy_bar_docs_large.npy', topics_file='data/true_topics.npy', num_models=1, subset_docs=5000, **data_config)
    true_ape_validation_generator = data.DataLoader(true_ape_validation_set, **loader_config)

    # pyro_scheduler is for ape_vae
    pyro_scheduler = CosineAnnealingWarmRestarts({'optimizer': torch.optim.Adam, 'T_0': 5000, 'optim_args': {"lr": .005}})
    vae_pyro_scheduler = CosineAnnealingWarmRestarts({'optimizer': torch.optim.Adam, 'T_0': 500, 'optim_args': {"lr": .00001}})
    ape_pyro_scheduler = CosineAnnealingWarmRestarts({'optimizer': torch.optim.Adam, 'T_0': 500, 'optim_args': {"lr": .005}})

    losses_to_record = {}

    models = ['avitm', 'nvdm']
    architectures = ['template', 'template_unnorm', 'pseudo_inverse', 'pseudo_inverse_unnorm', 'pseudo_inverse_scaled']

    # test APE, no training
    ape_model_config = deepcopy(model_config)
    for combo in itertools.product(['true_topics', 'random_topics'], models, architectures):
        topic_type, model_type, architecture = combo
        ape_model_config['model_type'] = model_type
        ape_model_config['architecture'] = architecture
        ape_model_config['n_hidden_layers'] = 0
        ape_model_config['n_hidden_units'] = ape_model_config['n_topics']

        if topic_type == 'true_topics':
            val_gen = true_ape_validation_generator
        elif topic_type == 'random_topics':
            val_gen = ape_validation_generator

        ape_vae = APE(**ape_model_config)
        ape_avi = TimedAVI(ape_vae.model, ape_vae.encoder_guide, ape_pyro_scheduler, loss=Trace_ELBO(), num_samples=100, encoder=ape_vae.encoder)
        val_loss = get_val_loss(ape_avi, val_gen, use_cuda, device)
        print(combo, val_loss)
        losses_to_record['.'.join(combo)] = val_loss
    wandb.log(losses_to_record)

    # test APE_VAE with training
    ape_vae_model_config = deepcopy(model_config)
    for combo in itertools.product(models, architectures):
        model_type, architecture = combo
        ape_vae_model_config['model_type'] = model_type
        ape_vae_model_config['architecture'] = architecture

        ape_vae = APE_VAE(**ape_vae_model_config)
        ape_vae_avi = TimedAVI(ape_vae.model, ape_vae.encoder_guide, pyro_scheduler, loss=Trace_ELBO(), num_samples=100, encoder=ape_vae.encoder)
        name = 'ape_vae_' + '_'.join(combo)
        ape_vae_avi = train_from_scratch(ape_vae_avi, training_generator, validation_generator, pyro_scheduler, name=name, **train_config)
        torch.save(ape_vae.state_dict(), os.path.join(args.results_dir, 'ape_vae.dict'))
        print('ape_vae finished')
        del ape_vae
        del ape_vae_avi

    # train VAE from scratch
    standard_model_config = deepcopy(model_config)
    standard_model_config['architecture'] = 'standard'
    vae = APE_VAE(**standard_model_config)
    vae_train_config = deepcopy(train_config)
    vae_train_config['epochs'] = 1500
    vae_avi = TimedAVI(vae.model, vae.encoder_guide, vae_pyro_scheduler, loss=Trace_ELBO(), num_samples=100, encoder=vae.encoder)
    vae_avi = train_from_scratch(vae_avi, training_generator, validation_generator, vae_pyro_scheduler, name='vae', **vae_train_config)
    torch.save(vae.state_dict(), os.path.join(args.results_dir, 'vae.dict'))
    print('vae finished')
    del vae
    del vae_avi

    # test APE with training, use only the true topics
    ape_vae_model_config = deepcopy(model_config)
    for combo in itertools.product(['avitm', 'nvdm'], ['template_unnorm', 'pseudo_inverse', 'pseudo_inverse_scaled']):
        model_type, architecture = combo
        ape_vae_model_config['model_type'] = model_type
        ape_vae_model_config['architecture'] = architecture

        ape = APE(**ape_vae_model_config)
        ape_avi = TimedAVI(ape.model, ape.encoder_guide, ape_pyro_scheduler, loss=Trace_ELBO(), num_samples=100, encoder=ape.encoder)
        ape_train_config = deepcopy(train_config)
        ape_train_config['epochs'] = 1
        ape_avi = train(ape_avi, true_ape_training_generator, true_ape_validation_generator, ape_pyro_scheduler, name='ape', **ape_train_config)
        torch.save(ape.state_dict(), os.path.join(args.results_dir, 'ape.dict'))
        print('ape finished')
