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

# this only differs from run.py in that the decoder is fixed
# as a result, the ordering of inference is slightly different since we don't have to rerun
# inference multiple times for SVI and HMC
# global params
topic_init = 'resources/topics_10x10.npy'
topic_fixed = True
datadir = 'toy_bars_10x10'
vocab_size = 100
n_topics = 18
results_dir = 'dump_fixed2'
results_file = 'results_svi_fixed.csv'

# vae params
vae_params = {
    # 'n_hidden_layers': [1, 2, 5, 10, 20],
    'n_hidden_layers': [1],
    'n_hidden_units': [10, 20, 50, 100, 200, 300, 500]

}
model = 'models.lda_lognorm'
mod = importlib.import_module(model)

# inference params
inference_techniques = ['vae', 'svi', 'hmc']
num_examples = 10

def train_save_VAE(n_hidden_layers, n_hidden_units):
    vae = mod.VAE_tf(n_hidden_layers=n_hidden_layers, n_hidden_units=n_hidden_units, n_topics=n_topics,
                     vocab_size=vocab_size, tensorboard=True, topic_init=topic_init, topic_fixed=topic_fixed)
    vae = train(data_tr, vae, training_epochs=100, tensorboard=True,
                tensorboard_logs_dir=results_dir + '/logs_{}_{}'.format(n_hidden_layers, n_hidden_units))
    vae.save(results_dir)
    vae.sess.close()
    tf.reset_default_graph()


def save_loglik_to_csv(data, model, posterior, num_samples=10):
    posterior_predictive = TracePredictive(vae.model, posterior, num_samples=num_samples)
    posterior_predictive_traces = posterior_predictive.run(data)
    # get the posterior predictive log likelihood
    posterior_predictive_density = evaluate_log_predictive_density(posterior_predictive_traces)
    posterior_predictive_density = float(posterior_predictive_density.detach().numpy())
    # columns: inference, model, dataset, n_hidden_layers, n_hidden_units, posterior_predictive_density
    with open(os.path.join(results_dir, results_file), 'a') as f:
        row = [inference_name, model, data_name, n_hidden_layers, n_hidden_units, posterior_predictive_density]
        csv_writer = csv.writer(f)
        csv_writer.writerow(row)

# 0. load data
data_tr, data_va, data_te = load_toy_bars(datadir, vocab_size)

# 1. train vae
# for n_hidden_layers in vae_params['n_hidden_layers']:
#     for n_hidden_units in vae_params['n_hidden_units']:
#         train_save_VAE(n_hidden_layers, n_hidden_units)

# 2. infer from model and evaluate
# get the indices for sample docs that will be consistent across runs
random_docs_idx = np.random.randint(0, 1000, size=num_examples)
# mcmc can't handle too much data
data_tr_small = data_tr[:1000]

for n_hidden_layers in vae_params['n_hidden_layers']:
    for n_hidden_units in vae_params['n_hidden_units']:
        # vae = mod.VAE_pyro(n_hidden_layers=n_hidden_layers, n_hidden_units=n_hidden_units,
        #                    n_topics=n_topics, vocab_size=vocab_size, topic_init=topic_init, topic_fixed=topic_fixed)
        vae = mod.VAE_pyro(n_hidden_layers=1, n_hidden_units=10,
                           n_topics=n_topics, vocab_size=vocab_size, topic_init=topic_init, topic_fixed=topic_fixed)
        state_dict = vae.load(results_dir)
        vae.load_state_dict(state_dict)
        for data_name, data in zip(['train', 'valid', 'te'], [data_tr_small, data_va, data_te]):
        # for data_name, data in zip(['test'], [data_te]):
            # instantiate the inference methods
            # pyro scheduler is used for both VAE and standard SVI, but it doesn't have any effect in the VAE case since
            # we never take any optimization steps
            pyro_scheduler = StepLR(
                {'optimizer': torch.optim.Adam, 'optim_args': {"lr": .1}, 'step_size': 10000, 'gamma': 0.95})
            # vae_svi = SVI(vae.model, vae.encoder_guide, pyro_scheduler, loss=Trace_ELBO(), num_steps=0)
            svi = SVI(vae.model, vae.mean_field_guide, pyro_scheduler, loss=TraceMeanField_ELBO(), num_steps=100, num_samples=1000)
            # mcmc = MCMC(NUTS(vae.model, adapt_step_size=True), num_samples=1000, warmup_steps=50)
            # mcmc_lda = MCMC(NUTS(vae.lda_model, adapt_step_size=True), num_samples=1000, warmup_steps=50)

            # get the sample docs for reconstruction
            n_docs = len(data)
            sample_docs = data[random_docs_idx]
            image = [sample_docs]
            data = torch.from_numpy(data.astype(np.float32))

            # for inference_name, inference in zip(['vae', 'svi', 'mcmc'], [vae_svi, svi, mcmc]):
            for inference_name, inference in zip(['svi'], [svi]):
                # hack to avoid rerunning SVI and MCMC
                # if (inference_name in ['svi', 'mcmc'] and n_hidden_layers != vae_params['n_hidden_layers'][0]
                #     and n_hidden_units != vae_params['n_hidden_layers'][0]):
                #     continue
                # hack to run mcmc only once
                if (inference_name == 'mcmc' and n_hidden_units > 10):
                    continue
                posterior = inference.run(data)
                for i in range(10):
                    save_loglik_to_csv(data, vae.model, posterior, num_samples=10)
                # reconstruct the data
                reconstructions = reconstruct_data(posterior, vae)
                # save sample reconstructions
                averaged_reconstructions = np.mean(reconstructions[:, random_docs_idx], axis=0)
                image.append(averaged_reconstructions)
            plot_name = os.path.join(results_dir, '_'.join([model, data_name, str(n_hidden_layers), str(n_hidden_units)]) + '.pdf')
            num_rows = len(image)
            image = np.array(image).reshape(num_rows * num_examples, vocab_size)
            plot_side_by_side_docs(image, plot_name, ncols=num_examples)




# 0. define VAE params and sample_idx
# 1. train and save all VAE models to get encoder
# 2. run inference given fixed decoder
# 2a. run the SVI and HMC baselines
# 2b. run the VAE across all combinations. Append the appropriate baseline images that we care about.