import os
import shutil
import itertools
import time
import pickle
import numpy as np
import math
import torch
import argparse
import matplotlib.pyplot as plt
from functools import partial
import logging
import psutil

from pyro.optim import StepLR
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from pyro.infer.mcmc import MCMC, NUTS
import pyro
import tensorflow as tf

from data.topics import toy_bars, permuted_toy_bars, diagonal_bars, generate_topics
from data.documents import generate_documents
from models.lda_meta import VAE_pyro, VAE_tf
from common import (
    train_save_VAE, save_speed_to_csv, save_loglik_to_csv, save_reconstruction_array, save_kl_to_csv, get_elbo_csv, run_posterior_evaluation
)
from visualization.reconstructions import plot_side_by_side_docs, plot_saved_samples
from visualization.posterior import plot_posterior, plot_posterior_v2, plot_posterior_v3
from visualization.ranking import plot_svi_vs_vae_elbo
from utils import softmax, unzip_X_and_topics, normalize1d
from training.train_vae import find_lr

parser = argparse.ArgumentParser(description='Results summary')
parser.add_argument('results_dir', type=str, help='directory of results')
parser.add_argument('--find_lr', help='find best learning rate', action='store_true')
parser.add_argument('--evaluate_svi_convergence', help='run SVI to see if it converges', action='store_true')
parser.add_argument('--evaluate_svi_convergence_with_vae_init', help='run SVI to see if it converges', action='store_true')
parser.add_argument('--train', help='train the model in tf', action='store_true')
parser.add_argument('--evaluate', help='evaluate posteriors in pyro', action='store_true')
parser.add_argument('--run_mcmc', help='run mcmc', action='store_true')
parser.add_argument('--use_cached', help='run mcmc', action='store_true')
parser.add_argument('--mdreviews', help='run mdreviews data', action='store_true')
parser.add_argument('--additive_topics', help='run mdreviews data', action='store_true')
args = parser.parse_args()

# where to write the results
results_dir = args.results_dir
results_file = 'results.csv'
print(results_dir)
if not os.path.exists(results_dir):
    os.system('mkdir -p ' + results_dir)
shutil.copy(os.path.abspath(__file__), os.path.join(results_dir, 'run_simple.py'))

sample_idx = list(range(10, 20, 2)) + list(range(90, 100, 2))
num_documents = 500
num_train_topics = 50
num_valid_topics = 10
num_test_topics = 10
num_combinations_to_evaluate = 300
random_topics_idx = 2

# global params
if args.mdreviews:
    n_topics = 100
    vocab_size = 7729
else:
    n_topics = 20
    vocab_size = 100

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(results_dir, 'memory_consumption.log'), when='D', interval=2)
formatter = logging.Formatter('%(asctime)s: %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

def get_memory_consumption():
    process = psutil.Process(os.getpid())
    logger.info('{}'.format(process.memory_info()[0]))

if args.run_mcmc:
    inference_names = ['mcmc', 'svi', 'svi_warmstart', 'vae', 'vae_single']
else:
    inference_names = ['svi', 'svi_warmstart', 'vae', 'vae_single']

model_config = {
    'vocab_size': vocab_size,
    'n_topics': n_topics,
    'results_dir': results_dir,
    'results_file': results_file,
    'inference': 'vae',
    'model_name': 'lda_meta',
    'architecture': 'naive',
    'scale_trainable': True,
    'n_hidden_layers': 2,
    'n_hidden_units': 100,
    'n_samples': 100,
    'decay_rate': .5,
    'decay_steps': 1000,
    'starting_learning_rate': .01,
    'n_steps_enc': 1,
    'custom_lr': False,
    'use_dropout': True,
    'use_adamw': False,
    'alpha': .01,
    'scale_type': 'mean',
    'tot_epochs': 800,
    'batch_size': 200,
    'seed': 0
}

model_config_single = model_config.copy()
model_config_single.update({'model_name': 'lda_orig', 'architecture': 'standard', 'n_hidden_layers': 5})
model_config_single.update({'starting_learning_rate': .01, 'tot_epochs': 400, 'batch_size': 150, 'decay_steps': 800})

# toy bars data
if args.use_cached and os.path.exists(os.path.join(results_dir, 'train_topics.npy')):
    logging.info('Loading data')
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
    logging.info('Creating data')
    if args.mdreviews:
        betas = []
        orig_topics = np.load('resources/mdreviews_topics3.npy')
        for i, topic in enumerate(orig_topics):
            beta = np.ones(vocab_size)
            # we take the top 100 words as the popular words
            popular_words = np.argpartition(topic, -100)[-100:]
            beta[popular_words] = 1000
            beta = normalize1d(beta)
            beta[popular_words] *= 50
            betas.append(beta)
    elif args.additive_topics:
        betas = []
        for i in range(n_topics):
            beta = np.ones(vocab_size)
            dim = math.sqrt(vocab_size)
            if i < dim:
                popular_words = [idx for idx in range(vocab_size) if idx % dim == i]
            else:
                popular_words = [idx for idx in range(vocab_size) if int(idx / dim) == i - dim]
            random_additions = list(np.random.choice(range(vocab_size), 20))
            beta[popular_words] = 1000
            beta[random_additions] = 200
            beta = normalize1d(beta)
            beta[popular_words] *= 5
            beta[random_additions] *= 2
            betas.append(beta)
    else:
        betas = []
        for i in range(n_topics):
            beta = np.ones(vocab_size)
            dim = math.sqrt(vocab_size)
            if i < dim:
                popular_words = [idx for idx in range(vocab_size) if idx % dim == i]
            else:
                popular_words = [idx for idx in range(vocab_size) if int(idx / dim) == i - dim]
            beta[popular_words] = 1000
            beta = normalize1d(beta)
            beta[popular_words] *= 5
            betas.append(beta)
    train_topics = generate_topics(n=num_train_topics, betas=betas, seed=0)
    valid_topics = generate_topics(n=num_valid_topics, betas=betas, seed=1)
    test_topics = generate_topics(n=num_test_topics, betas=betas, seed=2)

    for i, topics in zip(range(10), train_topics):
        plot_side_by_side_docs(topics, os.path.join(results_dir, 'train_topics_{}.pdf'.format(str(i).zfill(3))))
    for i, topics in enumerate(valid_topics):
        plot_side_by_side_docs(topics, os.path.join(results_dir, 'valid_topics_{}.pdf'.format(str(i).zfill(3))))
    for i, topics in enumerate(test_topics):
        plot_side_by_side_docs(topics, os.path.join(results_dir, 'test_topics_{}.pdf'.format(str(i).zfill(3))))

    documents, doc_topic_dists = generate_documents(train_topics[0], num_documents, alpha=.05, seed=0)
    plot_side_by_side_docs(documents[:40], os.path.join(results_dir, 'documents.pdf'))

    np.save(os.path.join(results_dir, 'train_topics.npy'), train_topics)
    np.save(os.path.join(results_dir, 'valid_topics.npy'), valid_topics)
    np.save(os.path.join(results_dir, 'test_topics.npy'), test_topics)
    np.save(os.path.join(results_dir, 'documents.npy'), documents)

logging.info('Data acquired')
get_memory_consumption()

train = (documents, train_topics)
valid = (documents, valid_topics)
test = (documents, test_topics)

def generate_datasets(train, valid, test, n):
    datasets = [train, valid, test]
    for dataset in datasets:
        docs, topics = dataset
        subset = [comb for _, comb in zip(range(n), itertools.product(docs, topics))]
        yield subset

# TODO: fix to evaluate on same number of topics
datasets = generate_datasets(train, valid, test, n=num_combinations_to_evaluate)
dataset_names = ['train', 'valid', 'test']

logging.info('Datasets generated')
get_memory_consumption()

# train the VAE and save the weights
if args.find_lr:
    def plot_lr(topics, model_config, name='learning_rates.png'):
        model_config['num_batches'] = math.ceil(len(documents) * len(topics) / model_config['batch_size'])
        vae = VAE_tf(test_lr=True, **model_config)
        train_data = list(itertools.product(documents, topics))
        log_lrs, losses = find_lr(vae, train_data, batch_size=model_config['batch_size'], final_value=1e2)
        print(log_lrs)
        print(losses)
        plt.plot(log_lrs[10:-5],losses[10:-5])
        plt.savefig(os.path.join(results_dir, name))
        plt.close()
        vae.sess.close()
        tf.reset_default_graph()
    
    plot_lr(train_topics, model_config)
    plot_lr([train_topics[random_topics_idx]], model_config_single, 'learning_rates_vae_single.png')

if args.evaluate_svi_convergence:
    vae = VAE_pyro(**model_config)
    num_steps = 600
    pyro_scheduler = StepLR({'optimizer': torch.optim.Adam, 'optim_args': {"lr": .1}, 'step_size': 10000, 'gamma': 0.95})
    svi = SVI(vae.model, vae.mean_field_guide, pyro_scheduler, loss=Trace_ELBO(), num_steps=num_steps, num_samples=100)
    data, topics = unzip_X_and_topics(next(datasets))
    data = torch.from_numpy(data.astype(np.float32))
    topics = torch.from_numpy(topics.astype(np.float32))
    losses = []
    for i in range(num_steps):
        loss = svi.step(data, topics)
        losses.append(loss)
    print(losses)
    plt.plot(range(num_steps), losses)
    plt.savefig(os.path.join(results_dir, 'svi_convergence.png'))

if args.evaluate_svi_convergence_with_vae_init:
    vae = VAE_pyro(**model_config)
    data, topics = unzip_X_and_topics(next(datasets))
    data = torch.from_numpy(data.astype(np.float32))
    topics = torch.from_numpy(topics.astype(np.float32))
    data_name = 'train'
    num_steps = 600
    model_config.update({
            'data_name': data_name,
            'results_dir': results_dir,
        })
    posterior_eval = partial(
            run_posterior_evaluation, 
            data=data, data_name=data_name, topics=topics, vae=vae, sample_idx=sample_idx, model_config=model_config)
    pyro_scheduler = StepLR({'optimizer': torch.optim.Adam, 'optim_args': {"lr": .1}, 'step_size': 10000, 'gamma': 0.95})
    vae_svi = SVI(vae.model, vae.encoder_guide, pyro_scheduler, loss=Trace_ELBO(), num_steps=0, num_samples=100)
    vae_svi = posterior_eval(vae_svi, 'vae')
    svi = SVI(vae.model, vae.mean_field_guide, pyro_scheduler, loss=Trace_ELBO(), num_steps=num_steps, num_samples=100)
    z_loc, z_scale = vae.encoder.forward(data, topics)
    print(z_loc[0])
    print(z_scale[0])
    pyro.clear_param_store()
    pyro.get_param_store().get_param('z_loc', init_tensor=z_loc.detach())
    pyro.get_param_store().get_param('z_scale', init_tensor=z_scale.detach())
    print(pyro.get_param_store().get_all_param_names())
    # pyro.get_param_store().replace_param("z_loc", z_loc, torch.zeros(data.shape[0], topics.shape[1]))
    # pyro.get_param_store().replace_param("z_scale", z_scale, torch.ones(data.shape[0], topics.shape[1]))
    losses = []
    for i in range(num_steps):
        print(pyro.get_param_store().get_param('z_loc')[0])
        print(pyro.get_param_store().get_param('z_scale')[0])
        loss = svi.step(data, topics)
        losses.append(loss)
        break
    print(losses)
    plt.plot(range(num_steps), losses)
    plt.savefig(os.path.join(results_dir, 'svi_convergence_vae_init.png'))

if args.train:
    logging.info('Starting train')
    train_save_VAE(
        train, valid, model_config,
        training_epochs=model_config['tot_epochs'], batch_size=model_config['batch_size'],
        hallucinations=False, tensorboard=True, shuffle=True, display_step=1,
        n_topics=n_topics, vocab_size=vocab_size, recreate_docs=False)

    logging.info('Finished train')
    logging.info('Starting training single')
    single_train = (documents, [train_topics[random_topics_idx]])
    train_save_VAE(
        single_train, valid, model_config_single,
        training_epochs=model_config_single['tot_epochs'], batch_size=model_config_single['batch_size'],
        hallucinations=False, tensorboard=True, shuffle=True, display_step=1,
        n_topics=n_topics, vocab_size=vocab_size, recreate_docs=False)
    logging.info('Finished training single')
# load the VAE into pyro for evaluation
if args.evaluate:
    vae = VAE_pyro(**model_config)
    state_dict = vae.load()
    vae.load_state_dict(state_dict)

    vae_single = VAE_pyro(**model_config_single)
    state_dict = vae_single.load()
    vae_single.load_state_dict(state_dict)

    if model_config['scale_type'] == 'mean':
        print(vae.encoder.scale)
    elif model_config['scale_type'] == 'sample':
        print(vae.decoder.scale)

    for data_name, data_and_topics in zip(dataset_names, datasets):
        data, topics = unzip_X_and_topics(data_and_topics)
        data = torch.from_numpy(data.astype(np.float32))
        topics = torch.from_numpy(topics.astype(np.float32))
        model_config.update({
            'data_name': data_name,
            'results_dir': results_dir,
        })
        model_config_single.update({
            'data_name': data_name,
            'results_dir': results_dir,
        })

        posterior_eval = partial(
            run_posterior_evaluation,
            data=data, data_name=data_name, topics=topics, vae=vae, sample_idx=sample_idx, model_config=model_config)
        single_vae_posterior_eval = partial(
            run_posterior_evaluation,
            data=data, data_name=data_name, topics=topics, vae=vae_single, sample_idx=sample_idx, model_config=model_config_single)

        pyro_scheduler = StepLR({'optimizer': torch.optim.Adam, 'optim_args': {"lr": .1}, 'step_size': 10000, 'gamma': 0.95})
        # Note: pyro scheduler doesn't have any effect in the VAE case since we never take any optimization steps
        logging.info('Starting VAE evaluation')
        pyro.clear_param_store()
        vae_svi = SVI(vae.model, vae.encoder_guide, pyro_scheduler, loss=Trace_ELBO(), num_steps=0, num_samples=100)
        vae_svi = posterior_eval(vae_svi, 'vae')
        get_memory_consumption()
        logging.info('Deleting vae_svi')
        del vae_svi
        get_memory_consumption()

        logging.info('Starting VAE single evaluation')
        pyro.clear_param_store()
        vae_svi_single = SVI(vae_single.model, vae_single.encoder_guide, pyro_scheduler, loss=Trace_ELBO(), num_steps=0, num_samples=100)
        vae_svi_single = single_vae_posterior_eval(vae_svi_single, 'vae_single')
        get_memory_consumption()
        logging.info('Deleting vae_svi_single')
        del vae_svi_single
        get_memory_consumption()

        logging.info('Starting SVI warmstart evaluation')
        pyro.clear_param_store()
        svi_warmstart = SVI(vae.model, vae.mean_field_guide, pyro_scheduler, loss=Trace_ELBO(), num_steps=600, num_samples=100)
        z_loc, z_scale = vae_single.encoder.forward(data, topics)
        pyro.clear_param_store()
        pyro.get_param_store().get_param('z_loc', init_tensor=z_loc.detach())
        pyro.get_param_store().get_param('z_scale', init_tensor=z_scale.detach())
        svi_warmstart = posterior_eval(svi_warmstart, 'svi_warmstart')
        get_memory_consumption()
        logging.info('Deleting svi_warmstart')
        del svi_warmstart
        get_memory_consumption()

        logging.info('Starting SVI evaluation')
        pyro.clear_param_store()
        svi = SVI(vae.model, vae.mean_field_guide, pyro_scheduler, loss=Trace_ELBO(), num_steps=600, num_samples=100)
        svi = posterior_eval(svi, 'svi')
        if args.run_mcmc:
            logging.info('Starting MCMC evaluation')
            nuts_kernel = NUTS(vae.model, adapt_step_size=True)
            nuts_kernel.initial_trace = svi.exec_traces[-1]
            get_memory_consumption()
            logging.info('Deleting svi')
            del svi
            get_memory_consumption()
            pyro.clear_param_store()
            mcmc = MCMC(nuts_kernel, num_samples=100, warmup_steps=100)
            mcmc = posterior_eval(mcmc, 'mcmc')
        else:
            get_memory_consumption()
            logging.info('Deleting svi')
            del svi
            get_memory_consumption()

        # save kl between MCMC and the others
        # if args.run_mcmc:
        #     save_kl_to_csv(results_dir, data_name)

    # plot the posteriors
    logging.info('Plotting posteriors')
    if model_config['scale_type'] == 'sample':
        scale = vae.decoder.scale.data.numpy()
    else:
        scale = 1
    # for i in sample_idx:
    #     plot_posterior(results_dir, i, dataset_names, inference_names, scale=scale)
    # plot_posterior_v2(results_dir, sample_idx, ['train', 'valid', 'test'], inference_names, scale)
    plot_posterior_v3(results_dir, sample_idx, ['train', 'valid', 'test'], inference_names, scale)

    # plot the reconstructions
    logging.info('Plotting reconstructions')
    datasets = generate_datasets(train, valid, test, n=num_combinations_to_evaluate)
    for data_name, data_and_topics in zip(dataset_names, datasets):
        data, _ = unzip_X_and_topics(data_and_topics)
        filenames = []
        for inference in inference_names:
            if inference == 'vae_single':
                file = '_'.join([inference, model_config_single['model_name'], data_name, str(model_config_single['n_hidden_layers']), str(model_config_single['n_hidden_units'])]) + '.npy'
            else:
                file = '_'.join([inference, model_config['model_name'], data_name, str(model_config['n_hidden_layers']), str(model_config['n_hidden_units'])]) + '.npy'
            filepath = os.path.join(os.getcwd(), results_dir, file)
            if os.path.exists(filepath):
                filenames.append(filepath)
            else:
                print(inference)

        plot_name = os.path.join(results_dir, data_name + '_vae_reconstructions.pdf')
        plot_saved_samples(np.array(data)[sample_idx], filenames, plot_name, vocab_size=vocab_size, intensity=10)
    
    logging.info('Plotting SVI vs VAE ELBOs')
    pyro.clear_param_store()
    # reload vae to be able to take in different-sized batches
    vae = VAE_pyro(**model_config)
    state_dict = vae.load()
    vae.load_state_dict(state_dict)
    get_elbo_csv(vae, results_dir)
    plot_svi_vs_vae_elbo(results_dir)