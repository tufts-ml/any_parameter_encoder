import argparse
import torch
from torch.utils import data
from pyro.optim import ExponentialLR

from pyro.infer import SVI, Trace_ELBO, NUTS, MCMC

from dataset import ToyBarsDataset
from model import VAE
from train import train

from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

parser = argparse.ArgumentParser(description='Results summary')
parser.add_argument('results_dir', type=str, help='directory of results')
parser.add_argument('architecture', type=str, help='encoder architecture')
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
}

data_config = {
    'doc_file': 'data/toy_bar_docs.npy',
    'num_models': 5,
    'n_topics': 20,
    'vocab_size': 100,
    'alpha': .1,
    'use_cuda': use_cuda
}

loader_config = {
    'batch_size': 64,
    'shuffle': True,
    'num_workers': 6}        

train_config = {
    'epochs': 2,
    'use_cuda': use_cuda,
    'results_dir': args.results_dir,
}

if __name__ == "__main__":
    vae = VAE(**model_config)
    pyro_scheduler = ExponentialLR({'optimizer': torch.optim.Adam, 'optim_args': {"lr": .01}, 'gamma': 0.95})
    vae_svi = SVI(vae.model, vae.encoder_guide, pyro_scheduler, loss=Trace_ELBO(), num_samples=100)
    training_set = ToyBarsDataset(training=True, **data_config)
    validation_set = ToyBarsDataset(training=False, **data_config)
    training_generator = data.DataLoader(training_set, **loader_config)
    validation_generator = data.DataLoader(validation_set, **loader_config)
    vae_svi = train(vae_svi, training_generator, validation_generator, **train_config)

    svi = SVI(vae_svi.model, vae.mean_field_guide, pyro_scheduler, loss=Trace_ELBO(), num_samples=100)
    nuts_kernel = NUTS(vae.model, adapt_step_size=True)
    mcmc = MCMC(nuts_kernel, num_samples=100, warmup_steps=100)

    evaluate_likelihood(vae_svi)
    evaluate_likelihood(svi)
    evaluate_likelihood(mcmc)