""" We look into the posteriors from different encoder architectures in comparison to SVI and MCMC """

import os
import argparse
import numpy as np
import itertools
import pickle
import torch
from torch.utils import data
import pyro
from pyro.optim import ExponentialLR, StepLR
from pyro.infer import Trace_ELBO, TraceMeanField_ELBO
from pyro.infer.mcmc import NUTS

from dataset import ToyBarsDataset
from model import APE
from train import train
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
parser.add_argument('--run_avi', help='run amortized variational inference', action='store_true')
parser.add_argument('--run_svi', help='run SVI', action='store_true')
parser.add_argument('--run_mcmc', help='run MCMC', action='store_true')
parser.add_argument('--warmstart_mcmc', help='warmstart MCMC', action='store_true')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
print(use_cuda)

model_config = {
    'n_hidden_units': 20,
    'n_hidden_layers': 0,
    'alpha': .1,
    'vocab_size': 100,
    'n_topics': 20,
    'use_cuda': use_cuda,
    'scale_type': 'sample',
    'skip_connections': False,
    'architecture': args.architecture
}

data_config = {
    'doc_file': 'data/toy_bar_docs_large.npy',
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
    'epochs': 2,
    'use_cuda': use_cuda,
}

eval_config = {
    'documents': 'data/toy_bar_docs.npy',
    'topics': 'data/true_topics.npy'
    # 'topics': 'data/train_topics.npy'
}

if __name__ == "__main__":
    wandb.init(sync_tensorboard=True, project="any_parameter_encoder", entity="lily", name=args.results_dir)
    names = []
    inferences = []

    vae = APE(**model_config)

    if args.run_avi:
        pyro_scheduler = ExponentialLR({'optimizer': torch.optim.Adam, 'optim_args': {"lr": .01}, 'gamma': 0.95})
        avi = TimedAVI(vae.model, vae.encoder_guide, pyro_scheduler, loss=TraceMeanField_ELBO(), num_samples=100, encoder=vae.encoder)
        names.append('avi')
        inferences.append(avi)

    if args.run_svi:
        # hyperparameters have been optimized
        pyro_scheduler = StepLR({'optimizer': torch.optim.Adam, 'optim_args': {"lr": .05}, 'step_size': 200, 'gamma': 0.95})
        print(pyro_scheduler)
        svi = TimedSVI(vae.model, vae.mean_field_guide, pyro_scheduler, loss=TraceMeanField_ELBO(), num_samples=100) #, num_steps=100000)
        training_set = ToyBarsDataset(training=True, topics_file='data/true_topics.npy', num_models=1, **data_config)
        training_generator = data.DataLoader(training_set, batch_size=500)
        n_epochs = 1000
        svi = train(svi, training_generator, training_generator, pyro_scheduler, **{'epochs': n_epochs, 'use_cuda': use_cuda, 'results_dir': args.results_dir})
        print(n_epochs)
        names.append('svi')
        inferences.append(svi)

    if args.run_mcmc:
        nuts_kernel = NUTS(vae.model, adapt_step_size=True)
        mcmc = TimedMCMC(nuts_kernel, num_samples=100, warmup_steps=100)
        names.append('mcmc')
        inferences.append(mcmc)

    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    all_docs = np.load(eval_config['documents'])[:5]
    all_topics = np.load(eval_config['topics'])
    np.save(os.path.join('debug', 'docs.npy'), all_docs)
    np.save(os.path.join('debug', 'topics.npy'), all_topics)
    # doc_idx = np.random.choice(range(len(all_docs)), size=300)
    # topic_idx = np.random.choice(range(len(all_topics)), size=300)
    # documents = all_docs[doc_idx]
    # topics = all_topics[topic_idx]
    if not os.path.exists('debug'):
        os.mkdir('debug')

    documents, topics = zip(*[combination for combination in itertools.product(all_docs, all_topics)])
    documents = np.array(documents)
    num_words = documents.sum()
    print('num_words', num_words)
    documents = torch.from_numpy(documents).type(dtype)
    topics = torch.from_numpy(np.array(topics)).type(dtype)

    for name, inference in zip(names, inferences):
        if args.warmstart_mcmc and isinstance(inference, TimedMCMC):
            # we re-initialize MCMC to use SVI posterior as a warm-start
            nuts_kernel.initial_trace = svi.exec_traces[-1]
            inference = TimedMCMC(nuts_kernel, num_samples=100, warmup_steps=100)
        posterior = inference.run(documents, topics)
        traces = []
        for tr in posterior.exec_traces:
            traces.append(tr.nodes['latent']['value'].detach().numpy())
        trace_filename = os.path.join('debug', name + '_' + args.architecture + '.npy')
        np.save(trace_filename, np.array(traces))
        likelihoods = []
        for _ in range(10):
            likelihood = get_posterior_predictive_density(documents, topics, vae.model, posterior)
            print('likelihood', likelihood)
            likelihoods.append(likelihood / num_words)
        print(name)
        print(np.mean(posterior.run_times), np.std(posterior.run_times))
        print(np.mean(likelihoods), np.std(likelihoods))
