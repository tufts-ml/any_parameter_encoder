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
parser.add_argument('--architecture', type=str, default='template', help='encoder architecture')
parser.add_argument('--run_avi', help='run amortized variational inference', action='store_true')
parser.add_argument('--run_svi', help='run SVI', action='store_true')
parser.add_argument('--run_mcmc', help='run MCMC', action='store_true')
parser.add_argument('--warmstart_mcmc', help='warmstart MCMC', action='store_true')
parser.add_argument('--local_testing', action='store_true')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
print("CUDA:", use_cuda)
if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

model_config = {
    'n_hidden_units': 100,
    'n_hidden_layers': 1,
    'results_dir': args.results_dir,
    'alpha': 0.1,
    'vocab_size': 25,
    'n_topics': 10,
    'use_cuda': use_cuda,
    'architecture': args.architecture,
    'scale_type': 'sample',
    'skip_connections': False,
}

data_config = {
    'doc_file': 'data/toy_bar_docs.npy',
    'n_topics': 10,
    'vocab_size': 25,
    'alpha': 0.1,
    'use_cuda': use_cuda,
    'num_docs': 2
}

loader_config = {
    'batch_size': 500,
    'shuffle': True,
    'num_workers': 0
}

train_config = {
    'epochs': 1,
    'use_cuda': use_cuda,
    'results_dir': args.results_dir,
}

eval_config = {
    'documents': 'data/toy_bar_docs.npy',
    'topics': 'data/toy_bar_docs_topics.npy',
    'n_trials': 10,
    'n_mc_samples': 50,
}

if __name__ == "__main__":
    # wandb.init(sync_tensorboard=True, project="any_parameter_encoder", entity="lily", name=args.results_dir)
    if args.local_testing:
        num_models = {'train': 5, 'val': 2, 'test': 2}
    else:
        num_models = {'train': 50000, 'val': 500, 'test': 500}
    names = []
    inferences = []

    vae = APE(**model_config)

    if args.run_avi:
        model_path = os.path.join(args.results_dir, 'ape.dict')
        if os.path.exists(model_path):
            device = torch.device("cuda:0" if use_cuda else "cpu")
            vae.load_state_dict(torch.load(model_path, map_location=device))

        pyro_scheduler = ExponentialLR({'optimizer': torch.optim.Adam, 'optim_args': {"lr": .01}, 'gamma': 0.95})
        avi = TimedAVI(vae.model, vae.encoder_guide, pyro_scheduler, loss=Trace_ELBO(), num_samples=100, encoder=vae.encoder)

        if not os.path.exists(model_path):
            training_set = ToyBarsDataset(topics_file='data/train_topics.npy', num_models=num_models['train'], **data_config)
            validation_set = ToyBarsDataset(topics_file='data/valid_topics.npy', num_models=num_models['val'], **data_config)
            test_set = ToyBarsDataset(topics_file='data/test_topics.npy', num_models=num_models['test'], **data_config)
            training_generator = data.DataLoader(training_set, **loader_config)
            validation_generator = data.DataLoader(validation_set, **loader_config)
            avi = train(avi, training_generator, validation_generator, pyro_scheduler, **train_config)

            # we only save the model to use in downstream inference
            torch.save(vae.state_dict(), model_path)

        names.append('avi')
        inferences.append(avi)

    if args.run_svi:
        # hyperparameters have been optimized
        pyro_scheduler = StepLR({'optimizer': torch.optim.Adam, 'optim_args': {"lr": .05}, 'step_size': 200, 'gamma': 0.95})
        svi = TimedSVI(vae.model, vae.mean_field_guide, pyro_scheduler, loss=Trace_ELBO(), num_samples=100) #, num_steps=100000)
        training_set = ToyBarsDataset(topics_file='data/test_topics.npy', num_models=num_models['test'], **data_config)
        training_generator = data.DataLoader(training_set, batch_size=500)
        if args.local_testing:
            n_epochs = 1
        else:
            n_epochs = 10000
        #svi = train(svi, training_generator, training_generator, pyro_scheduler, **{'epochs': n_epochs, 'use_cuda': use_cuda, 'results_dir': args.results_dir})
        names.append('svi')
        inferences.append(svi)

    if args.run_mcmc:
        nuts_kernel = NUTS(vae.model, adapt_step_size=True)
        mcmc = TimedMCMC(nuts_kernel, num_samples=100, warmup_steps=100)
        names.append('mcmc')
        inferences.append(mcmc)

    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    documents = np.load(eval_config['documents'])
    topics = np.load(eval_config['topics'])
    # doc_idx = np.random.choice(range(len(all_docs)), size=300)
    # topic_idx = np.random.choice(range(len(all_topics)), size=300)
    # documents = all_docs[doc_idx]
    # topics = all_topics[topic_idx]
    #documents, topics = zip(*[combination for combination in itertools.product(all_docs, all_topics)])
    #documents = np.array(documents)
    num_words = documents.sum()
    print('num_words', num_words)
    documents = torch.from_numpy(np.array(documents)).type(dtype)
    topics = torch.from_numpy(np.array(topics)).type(dtype)

    for name, inference in zip(names, inferences):
        if args.warmstart_mcmc and isinstance(inference, TimedMCMC):
            # we re-initialize MCMC to use SVI posterior as a warm-start
            nuts_kernel.initial_trace = svi.exec_traces[-1]
            inference = TimedMCMC(nuts_kernel, num_samples=100, warmup_steps=100)
        posterior = inference.run(documents, topics)
        logpmf_per_token_list = []
        for _ in range(eval_config['n_trials']):
            logpmf_per_tok = get_posterior_predictive_density(
                documents, topics, vae.model, posterior, num_samples=eval_config['n_mc_samples']) / num_words
            logpmf_per_token_list.append(logpmf_per_tok)
        print("%s after %d trials" % (name, eval_config['n_trials']))
        print("log pmf per token: % 6.2f (min % 6.2f, max % 6.2f)" % (
            np.mean(logpmf_per_token_list),
            np.min(logpmf_per_token_list),
            np.max(logpmf_per_token_list)))
        print("runtime: %6.4f sec (min %6.4f, max %6.4f)" % (
            np.mean(posterior.run_times), np.min(posterior.run_times), np.max(posterior.run_times)))


