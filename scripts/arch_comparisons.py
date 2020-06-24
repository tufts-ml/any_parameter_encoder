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
    project_name = "ape_debug"
    wandb.init(sync_tensorboard=True, project=project_name, entity="lily", name=args.results_dir, reinit=True)
    names = []
    inferences = []

    # true_ape_training_set = ToyBarsDataset(training=True, doc_file='data/toy_bar_docs_large.npy', topics_file='data/true_topics.npy', num_models=1, subset_docs=50000, **data_config)
    # true_ape_training_generator = data.DataLoader(true_ape_training_set, **loader_config)
    # training = True just gives us more data
    # true_ape_validation_set = NonToyBarsDataset(training=True, doc_file='data/non_toy_bars_docs.npy', topics_file='data/non_toy_bars_topics.npy', num_models=1, num_docs=5000, **data_config)
    # true_ape_validation_generator = data.DataLoader(true_ape_validation_set, **loader_config)
    toy_bars = ToyBarsDataset(training=True, doc_file='data/toy_bars/docs_many_words.npy', topics_file='data/toy_bars/topics_many_words.npy', num_models=1, num_docs=5000, avg_num_words=500, **data_config)
    toy_bars_gen = data.DataLoader(toy_bars, **loader_config)

    losses_to_record = {}

    models = ['avitm', 'nvdm']
    # architectures = ['template', 'template_unnorm', 'template_scaled', 'pseudo_inverse', 'pseudo_inverse_unnorm', 'pseudo_inverse_scaled']
    architectures = ['prior']

    # test APE, no training
    ape_model_config = deepcopy(model_config)
    values = []
    for seed in range(10):
        pyro.set_rng_seed(seed)
        for combo in itertools.product(['toy_bars'], models, architectures):
            for loss in [Trace_ELBO, TraceMeanField_ELBO]:
                # ape_pyro_scheduler = CosineAnnealingWarmRestarts({'optimizer': torch.optim.Adam, 'T_0': 500, 'optim_args': {"lr": .005}})
                ape_pyro_scheduler = StepLR({'optimizer': torch.optim.Adam, 'optim_args': {"lr": .01}, "step_size": 250, "gamma": .5})
                topic_type, model_type, architecture = combo
                ape_model_config['model_type'] = model_type
                ape_model_config['architecture'] = architecture
                ape_model_config['n_hidden_layers'] = 0
                ape_model_config['n_hidden_units'] = ape_model_config['n_topics']

                if topic_type == 'toy_bars':
                    val_gen = toy_bars_gen
                else:
                    raise NotImplementedError("unsupported topic_type")

                ape_vae = APE(**ape_model_config)
                ape_avi = TimedAVI(ape_vae.model, ape_vae.encoder_guide, ape_pyro_scheduler, loss=loss(), num_samples=100, encoder=ape_vae.encoder)
                val_loss = get_val_loss(ape_avi, val_gen, use_cuda, device, scaled=True)
                print(combo, val_loss)
                values.append([topic_type, model_type, architecture, loss.__name__, val_loss, seed])
                df = pd.DataFrame(values)
                df.columns = ['topic_type', 'model_type', 'architecture', 'metric', 'loss', 'seed']
                df.to_csv('no_training.csv')


    training_set = ToyBarsDocsDataset(training=True, doc_file='data/toy_bar_docs_large.npy', subset_docs=50000, **data_config)
    validation_set = ToyBarsDocsDataset(training=False, doc_file='data/toy_bar_docs_large.npy', subset_docs=50000, **data_config)
    training_generator = data.DataLoader(training_set, **loader_config)
    validation_generator = data.DataLoader(validation_set, **loader_config)

    # test APE_VAE with training
    ape_vae_model_config = deepcopy(model_config)
    for combo in itertools.product(models, architectures):
        # ape_vae_pyro_scheduler = CosineAnnealingWarmRestarts({'optimizer': torch.optim.Adam, 'T_0': 5000, 'optim_args': {"lr": .005}})
        ape_vae_pyro_scheduler = StepLR({'optimizer': torch.optim.Adam, 'optim_args': {"lr": .01}, "step_size": 250, "gamma": .5})
        model_type, architecture = combo
        ape_vae_model_config['model_type'] = model_type
        ape_vae_model_config['architecture'] = architecture
        name = 'ape_vae_' + "_".join(combo)
        wandb.init(sync_tensorboard=True, project=project_name, entity="lily", name=name, reinit=True)

        ape_vae = APE_VAE(**ape_vae_model_config)
        ape_vae_avi = TimedAVI(ape_vae.model, ape_vae.encoder_guide, ape_vae_pyro_scheduler, loss=Trace_ELBO(retain_graph=True), num_samples=100, encoder=ape_vae.encoder)
        ape_vae_avi = train_from_scratch(ape_vae_avi, training_generator, validation_generator, ape_vae_pyro_scheduler, name='', **train_config)
        # torch.save(ape_vae.state_dict(), os.path.join(args.results_dir, 'ape_vae.dict'))
        print('ape_vae finished')
        del ape_vae
        del ape_vae_avi
        wandb.join()

    # train VAE from scratch
    # vae_pyro_scheduler = CosineAnnealingWarmRestarts({'optimizer': torch.optim.Adam, 'T_0': 500, 'optim_args': {"lr": .00001}})
    vae_pyro_scheduler = StepLR({'optimizer': torch.optim.Adam, 'optim_args': {"lr": .01}, "step_size": 250, "gamma": .5})
    wandb.init(sync_tensorboard=True, project=project_name, entity="lily", name='vae', reinit=True)
    standard_model_config = deepcopy(model_config)
    standard_model_config['architecture'] = 'standard'
    vae = APE_VAE(**standard_model_config)
    vae_train_config = deepcopy(train_config)
    vae_train_config['epochs'] = 1500
    vae_avi = TimedAVI(vae.model, vae.encoder_guide, vae_pyro_scheduler, loss=Trace_ELBO(), num_samples=100, encoder=vae.encoder)
    vae_avi = train_from_scratch(vae_avi, training_generator, validation_generator, vae_pyro_scheduler, name='', **vae_train_config)
    torch.save(vae.state_dict(), os.path.join(args.results_dir, 'vae.dict'))
    print('vae finished')
    del vae
    del vae_avi
    wandb.join()

    import sys; sys.exit()
    # ape_training_set = ToyBarsDataset(training=True, doc_file='data/toy_bar_docs_large.npy', topics_file='data/train_topics.npy', num_models=20000, subset_docs=5000, **data_config)
    # ape_validation_set = ToyBarsDataset(training=False, doc_file='data/toy_bar_docs_large.npy', topics_file='data/valid_topics.npy', num_models=50, subset_docs=5000, **data_config)
    # ape_training_generator = data.DataLoader(ape_training_set, **loader_config)
    # ape_validation_generator = data.DataLoader(ape_validation_set, **loader_config)
    # # create the many_words docs dataset
    # ape_validation_set_many_words = ToyBarsDataset(training=False, doc_file='data/toy_bar_docs_large_many_words.npy', topics_file='data/valid_topics.npy', num_models=50, subset_docs=5000, avg_num_words=500, **data_config)
    # ape_validation_generator_many_words = data.DataLoader(ape_validation_set_many_words, **loader_config)

    # test APE with training, use only the true topics
    ape_vae_model_config = deepcopy(model_config)
    for combo in itertools.product(['avitm', 'nvdm'], ['template_unnorm', 'pseudo_inverse', 'pseudo_inverse_scaled']):
        # ape_pyro_scheduler = CosineAnnealingWarmRestarts({'optimizer': torch.optim.Adam, 'T_0': 500, 'optim_args': {"lr": .005}})
        ape_pyro_scheduler = StepLR({'optimizer': torch.optim.Adam, 'optim_args': {"lr": .01}, "step_size": 250, "gamma": .5})
        model_type, architecture = combo
        ape_vae_model_config['model_type'] = model_type
        ape_vae_model_config['architecture'] = architecture
        name = 'ape_true_' + "_".join(combo)
        wandb.init(sync_tensorboard=True, project=project_name, entity="lily", name=name, reinit=True)

        ape = APE(**ape_vae_model_config)
        ape_avi = TimedAVI(ape.model, ape.encoder_guide, ape_pyro_scheduler, loss=Trace_ELBO(), num_samples=100, encoder=ape.encoder)
        ape_train_config = deepcopy(train_config)
        ape_train_config['epochs'] = 100
        ape_avi = train(ape_avi, true_ape_training_generator, true_ape_validation_generator, ape_pyro_scheduler, name='', **ape_train_config)
        # torch.save(ape.state_dict(), os.path.join(args.results_dir, 'ape.dict'))
        print('ape finished')
        del ape
        del ape_avi
        wandb.join()
