import torch
from torch.utils import data
from pyro.optim import ExponentialLR

from pyro.infer import SVI, Trace_ELBO

from dataset import ToyBarsDataset
from model import VAE
from train import train


use_cuda = torch.cuda.is_available()

model_config = {
    'n_hidden_units': 100,
    'n_hidden_layers': 2,
    'model_name': None,
    'results_dir': None,
    'alpha': .1,
    'vocab_size': 100,
    'n_topics': 20,
    'use_cuda': use_cuda,
    'architecture': 'naive',
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

loader_config = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}

if __name__ == "__main__":
    vae = VAE(**model_config)
    pyro_scheduler = ExponentialLR({'optimizer': torch.optim.Adam, 'optim_args': {"lr": .01}, 'gamma': 0.95})
    vae_svi = SVI(vae.model, vae.encoder_guide, pyro_scheduler, loss=Trace_ELBO(), num_samples=100)
    training_set = ToyBarsDataset(**data_config)
    training_generator = data.DataLoader(training_set, **loader_config)
    train(vae_svi, training_generator, 10, use_cuda)
