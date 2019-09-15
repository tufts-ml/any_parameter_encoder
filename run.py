import os
import shutil
import itertools
import time
import pickle
import numpy as np
import math
import torch
import argparse

from pyro.optim import StepLR
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from pyro.infer.mcmc import MCMC, NUTS
import pyro
import tensorflow as tf

from data.topics import toy_bars, permuted_toy_bars, diagonal_bars, generate_topics
from data.documents import generate_documents
from models.lda_meta import VAE_pyro
from common import train_save_VAE, save_speed_to_csv, save_loglik_to_csv, save_reconstruction_array
from visualization.reconstructions import plot_side_by_side_docs, plot_saved_samples
from visualization.posterior import plot_posterior
from utils import softmax, unzip_X_and_topics, normalize1d

parser = argparse.ArgumentParser(description='Results summary')
parser.add_argument('results_dir', type=str, help='directory of results')
parser.add_argument('--train', help='train the model in tf', action='store_true')
parser.add_argument('--evaluate', help='evaluate posteriors in pyro', action='store_true')
parser.add_argument('--run_mcmc', help='run mcmc', action='store_true')
parser.add_argument('--use_cached', help='run mcmc', action='store_true')
args = parser.parse_args()

# where to write the results
results_dir = args.results_dir
results_file = 'results.csv'
print(results_dir)
if not os.path.exists(results_dir):
    os.system('mkdir -p ' + results_dir)
shutil.copy(os.path.abspath(__file__), os.path.join(results_dir, 'run_simple.py'))

# global params
n_topics = 20
vocab_size = 100
sample_idx = list(range(10))
# epochs = 1
epochs = 1000

model_config = {
    'vocab_size': vocab_size,
    'n_topics': n_topics,
    'results_dir': results_dir,
    'results_file': results_file,
    'inference': 'vae',
    'model_name': 'lda_meta',
    'architecture': 'naive',
    'scale_trainable': True,
    'n_hidden_layers': 1,
    'n_hidden_units': 100,
    'n_samples': 100,
    'decay_rate': .9,
    'decay_steps': 1000,
    'starting_learning_rate': .01,
    'n_steps_enc': 1,
    'custom_lr': False,
    'use_dropout': True,
    'use_adamw': False,
    'alpha': .01,
    'scale_type': 'mean'
}

# toy bars data
if args.use_cached and os.path.exists(os.path.join(results_dir, 'train_topics.npy')):
    train_topics = np.load(os.path.join(results_dir, 'train_topics.npy'))
    valid_topics = np.load(os.path.join(results_dir, 'valid_topics.npy'))
    test_topics = np.load(os.path.join(results_dir, 'test_topics.npy'))
    documents = np.load(os.path.join(results_dir, 'documents.npy'))
elif args.use_cached and os.path.exists(os.path.join('experiments', 'train_topics.npy')):
    train_topics = np.load(os.path.join('experiments', 'train_topics.npy'))
    valid_topics = np.load(os.path.join('experiments', 'valid_topics.npy'))
    test_topics = np.load(os.path.join('experiments', 'test_topics.npy'))
    documents = np.load(os.path.join('experiments', 'documents.npy'))
else:
    betas = []
    test_betas = []
    for i in range(n_topics):
        beta = np.ones(vocab_size)
        dim = math.sqrt(vocab_size)
        if i < dim:
            popular_words = [idx for idx in range(vocab_size) if idx % dim == i]
        else:
            popular_words = [idx for idx in range(vocab_size) if int(idx / dim) == i - dim]
        beta[popular_words] = 1000
        betas.append(normalize1d(beta))
        test_betas.append(normalize1d(beta + 1))
    train_topics = generate_topics(n=5, betas=betas, seed=0)
    valid_topics = generate_topics(n=5, betas=betas, seed=1)
    test_topics = generate_topics(n=5, betas=betas, seed=0)

    for i, topics in enumerate(train_topics):
        plot_side_by_side_docs(topics, os.path.join(results_dir, 'train_topics_{}.pdf'.format(str(i).zfill(3))))
    for i, topics in enumerate(valid_topics):
        plot_side_by_side_docs(topics, os.path.join(results_dir, 'valid_topics_{}.pdf'.format(str(i).zfill(3))))
    for i, topics in enumerate(test_topics):
        plot_side_by_side_docs(topics, os.path.join(results_dir, 'test_topics_{}.pdf'.format(str(i).zfill(3))))

    documents, doc_topic_dists = generate_documents(train_topics[0], 50, alpha=.01, seed=0)

    np.save(os.path.join(results_dir, 'train_topics.npy'), train_topics)
    np.save(os.path.join(results_dir, 'valid_topics.npy'), valid_topics)
    np.save(os.path.join(results_dir, 'test_topics.npy'), test_topics)
    np.save(os.path.join(results_dir, 'documents.npy'), documents)

# TODO: perform correct queuing so full dataset doesn't need to be in memory
train = list(itertools.product(documents, train_topics))
valid = list(itertools.product(documents, valid_topics))
test = list(itertools.product(documents, test_topics))
# TODO: fix to evaluate on same number of topics
datasets = [train[:300], valid[:300], test[:300]]
# datasets = [train, valid, test]
dataset_names = ['train', 'valid', 'test']

# train the VAE and save the weights
if args.train:
    vae = train_save_VAE(train, valid, model_config, training_epochs=epochs, batch_size=800, hallucinations=False, tensorboard=True, shuffle=True, display_step=1)
# load the VAE into pyro for evaluation
if args.evaluate:
    vae = VAE_pyro(**model_config)
    state_dict = vae.load()
    vae.load_state_dict(state_dict)
    if model_config['scale_type'] == 'mean':
        print(vae.encoder.scale)
    elif model_config['scale_type'] == 'sample':
        print(vae.decoder.scale)
    for data_name, data_and_topics in zip(dataset_names, datasets):
        data, topics = unzip_X_and_topics(data_and_topics)
        data = torch.from_numpy(data.astype(np.float32))
        topics = torch.from_numpy(topics.astype(np.float32))
        pyro_scheduler = StepLR({'optimizer': torch.optim.Adam, 'optim_args': {"lr": .1}, 'step_size': 10000, 'gamma': 0.95})
        # Note: pyro scheduler doesn't have any effect in the VAE case since we never take any optimization steps
        vae_svi = SVI(vae.model, vae.encoder_guide, pyro_scheduler, loss=Trace_ELBO(), num_steps=0, num_samples=100)
        svi = SVI(vae.model, vae.mean_field_guide, pyro_scheduler, loss=Trace_ELBO(), num_steps=400, num_samples=100)
        mcmc = MCMC(NUTS(vae.model, adapt_step_size=True), num_samples=100, warmup_steps=50)
        if args.run_mcmc:
            inference_methods = [vae_svi, svi, mcmc]
        else:
            inference_methods = [vae_svi, svi]
        for inference_name, inference in zip(['vae', 'svi', 'mcmc'], inference_methods):
            model_config.update({
                'data_name': data_name,
                'inference': inference_name,
                'results_dir': results_dir,
            })
            print(inference_name)
            print(model_config)

            start = time.time()
            posterior = inference.run(data, topics)
            end = time.time()
            save_speed_to_csv(model_config, end - start)

            save_reconstruction_array(vae, topics, posterior, sample_idx, model_config)

            # save the estimated posterior predictive log likelihood
            for i in range(10):
                # saves a separate row to the csv
                save_loglik_to_csv(data, topics, vae.model, posterior, model_config, num_samples=10)
            
            # save the posteriors for later analysis
            if inference_name == 'mcmc':
                # [num_samples, num_docs, num_topics]
                samples = [t.nodes['latent']['value'].detach().cpu().numpy() for t in posterior.exec_traces]
                np.save(os.path.join(results_dir, '{}_{}_samples.npy'.format(data_name, inference_name)), np.array(samples))
            else:
                if inference_name == 'vae':
                    z_loc, z_scale = vae.encoder.forward(data, topics)
                    z_loc = z_loc.data.numpy()
                    z_scale = z_scale.data.numpy()
                elif inference_name == 'svi':
                    z_loc = pyro.get_param_store().match('z_loc')['z_loc'].detach().numpy()
                    z_scale = pyro.get_param_store().match('z_scale')['z_scale'].detach().numpy()
                np.save(os.path.join(results_dir, '{}_{}_z_loc.npy'.format(data_name, inference_name)), z_loc)
                np.save(os.path.join(results_dir, '{}_{}_z_scale.npy'.format(data_name, inference_name)), z_scale)

            pyro.clear_param_store()
    # plot the posteriors
    for i in sample_idx:
        if args.run_mcmc:
            inference_names = ['vae', 'svi', 'mcmc']
        else:
            inference_names = ['vae', 'svi']
        if model_config['scale_type'] == 'sample':
            scale = vae.decoder.scale
        else:
            scale = 1
        plot_posterior(results_dir, i, dataset_names, inference_names, scale=scale)

    # plot the reconstructions
    for data_name, data_and_topics in zip(dataset_names, datasets):
        data, _ = unzip_X_and_topics(data_and_topics)
        filenames = []
        for inference in ['vae', 'svi', 'mcmc']:
            file = '_'.join([inference, model_config['model_name'], data_name, str(model_config['n_hidden_layers']), str(model_config['n_hidden_units'])]) + '.npy'
            filepath = os.path.join(os.getcwd(), results_dir, file)
            if os.path.exists(filepath):
                filenames.append(filepath)

        plot_name = os.path.join(results_dir, data_name + '_vae_reconstructions.pdf')
        plot_saved_samples(np.array(data)[sample_idx], filenames, plot_name, vocab_size=vocab_size, intensity=10)