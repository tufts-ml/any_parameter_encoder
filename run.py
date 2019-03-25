import os
import importlib
import csv
import numpy as np
import tensorflow as tf
import torch
from datasets.load import load_toy_bars
from training.train_vae import train

import pyro
from pyro.optim import StepLR
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from pyro.infer.abstract_infer import TracePredictive
from pyro.infer.mcmc import MCMC, NUTS
from evaluation.evaluate_posterior import evaluate_log_predictive_density
from evaluation.evaluate_posterior import reconstruct_data
from evaluation.evaluate_posterior import plot_side_by_side_docs

# global params
datadir = 'toy_bars_10x10'
vocab_size = 100
n_topics = 18
results_dir = 'dump'

# vae params
vae_params = {
    'n_hidden_layers': [1, 2, 5],
    'n_hidden_units': [10, 20, 50, 100]
}
model = 'models.lda_lognorm'
mod = importlib.import_module(model)

# inference params
inference_techniques = ['vae', 'svi', 'hmc']


def train_save_VAE(n_hidden_layers, n_hidden_units):
    vae = mod.VAE_tf(n_hidden_layers=n_hidden_layers, n_hidden_units=n_hidden_units, n_topics=n_topics,
                     vocab_size=vocab_size, tensorboard=True)
    vae = train(data_tr, vae, training_epochs=100, tensorboard=True,
                tensorboard_logs_dir='dump/logs_{}_{}'.format(n_hidden_layers, n_hidden_units))
    vae.save(results_dir)
    vae.sess.close()
    tf.reset_default_graph()


# 0. load data
data_tr, data_va, data_te = load_toy_bars(datadir, vocab_size)

# 1. train vae
# for n_hidden_layers in vae_params['n_hidden_layers']:
#     for n_hidden_units in vae_params['n_hidden_units']:
#         train_save_VAE(n_hidden_layers, n_hidden_units)

# 2. infer from model and evaluate
for n_hidden_layers in vae_params['n_hidden_layers']:
    for n_hidden_units in vae_params['n_hidden_units']:
        vae = mod.VAE_pyro(n_hidden_layers=n_hidden_layers, n_hidden_units=n_hidden_units, n_topics=n_topics, vocab_size=vocab_size)
        state_dict = vae.load(results_dir)
        vae.load_state_dict(state_dict)
        # instantiate the inference methods
        # pyro scheduler is used for both VAE and standard SVI, but it doesn't have any effect in the VAE case since
        # we never take any optimization steps
        pyro_scheduler = StepLR(
            {'optimizer': torch.optim.Adam, 'optim_args': {"lr": .1}, 'step_size': 1000, 'gamma': 0.9})
        vae_svi = SVI(vae.model, vae.encoder_guide, pyro_scheduler, loss=Trace_ELBO(), num_steps=0)
        svi = SVI(vae.model, vae.mean_field_guide, pyro_scheduler, loss=TraceMeanField_ELBO(), num_steps=30)
        mcmc = MCMC(NUTS(vae.model, adapt_step_size=True), num_samples=10, warmup_steps=500)
        for data_name, data in zip(['train', 'valid', 'test'], [data_tr, data_va, data_te]):
            data = torch.from_numpy(data.astype(np.float32))
            for inference_name, inference in zip(['vae', 'svi', 'mcmc'], [vae_svi, svi, mcmc]):
                posterior = inference.run(data)
                posterior_predictive = TracePredictive(vae.model, posterior, num_samples=10)
                posterior_predictive_traces = posterior_predictive.run(data)
                # get the posterior predictive log likelihood
                posterior_predictive_density = evaluate_log_predictive_density(posterior_predictive_traces)
                # columns: inference, model, dataset, n_hidden_layers, n_hidden_units, posterior_predictive_density
                with open('results.csv', 'a') as f:
                    row = [inference_name, model, data_name, n_hidden_layers, n_hidden_units, posterior_predictive_density]
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(row)
                # reconstruct the data
                reconstructions = reconstruct_data(posterior, vae)
                # save sample reconstructions
                reconstruction_name = "dump/" + '_'.join(model, n_hidden_layers, n_hidden_units, inference_name, data_name) + '.npy'
                np.save(reconstruction_name, reconstructions)

# 3. visualize