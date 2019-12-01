import os
import shutil
import itertools
import csv
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
from pyro.infer.abstract_infer import TracePredictive
from pyro.infer.mcmc import MCMC, NUTS
from pyro.util import torch_isnan
import pyro
import tensorflow as tf

from data.topics import toy_bars, permuted_toy_bars, diagonal_bars, generate_topics
from data.documents import generate_documents
from models.lda_meta import VAE_pyro, VAE_tf
from common import (
    train_save_VAE, save_speed_to_csv, save_loglik_to_csv, save_reconstruction_array, run_posterior_evaluation, get_elbo_csv
)
from visualization.reconstructions import plot_side_by_side_docs, plot_saved_samples
from visualization.posterior import plot_posterior, plot_posterior_v2, plot_posterior_v3
from visualization.ranking import plot_svi_vs_vae_elbo_v1
from visualization.times import plot_times
from evaluation.evaluate_posterior import evaluate_log_predictive_density
from utils import softmax, unzip_X_and_topics, normalize1d
from training.train_vae import find_lr

parser = argparse.ArgumentParser(description='Results summary')
parser.add_argument('results_dir', type=str, help='directory of results')
parser.add_argument('architecture', type=str, help='directory of results')
parser.add_argument('n_hidden_layers', type=int, help='directory of results')
parser.add_argument('n_hidden_units', type=int, help='directory of results')
parser.add_argument('--new_model_path', help='use model from ape repo', action='store_true')
parser.add_argument('--find_lr', help='find best learning rate', action='store_true')
parser.add_argument('--evaluate_svi_convergence', help='run SVI to see if it converges', action='store_true')
parser.add_argument('--evaluate_svi_convergence_with_vae_init', help='run SVI to see if it converges', action='store_true')
parser.add_argument('--train', help='train the model in tf', action='store_true')
parser.add_argument('--train_single', help='train the model in tf', action='store_true')
parser.add_argument('--evaluate', help='evaluate posteriors in pyro', action='store_true')
parser.add_argument('--run_standard_vae', help='run mcmc', action='store_true')
parser.add_argument('--run_svi', help='run mcmc', action='store_true')
parser.add_argument('--run_mcmc', help='run mcmc', action='store_true')
parser.add_argument('--use_cached', help='run mcmc', action='store_true')
parser.add_argument('--mdreviews', help='run mdreviews data', action='store_true')
parser.add_argument('--additive_topics', help='run mdreviews data', action='store_true')
parser.add_argument('--uniform_prior', help='run mdreviews data', action='store_true')
parser.add_argument('--plot', help='run mdreviews data', action='store_true')
args = parser.parse_args()

# where to write the results
results_dir = args.results_dir
results_file = 'results.csv'
print(results_dir)
if not os.path.exists(results_dir):
    os.system('mkdir -p ' + results_dir)
shutil.copy(os.path.abspath(__file__), os.path.join(results_dir, 'run_simple.py'))

# sample_idx = list(range(10, 20, 2)) + list(range(90, 100, 2))
sample_idx = [0, 1, 52, 53, 104, 105, 156, 157, 208, 209]
# sample_idx = list(range(10))
num_documents = 50
num_train_topics = 5
num_valid_topics = 10
num_test_topics = 10
num_combinations_to_evaluate = 300
random_topics_idx = 2
train_alpha = .1
valid_alpha = .05

# global params
if args.mdreviews:
    # n_topics = 100
    # vocab_size = 7729
    n_topics = 30
    vocab_size = 3000
else:
    n_topics = 20
    vocab_size = 100
    avg_num_words = 50

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(results_dir, 'memory_consumption.log'))
formatter = logging.Formatter('%(asctime)s: %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

def get_memory_consumption():
    process = psutil.Process(os.getpid())
    logger.info('{}'.format(process.memory_info()[0] / 1e9))

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
    'architecture': args.architecture,
    'scale_trainable': True,
    'n_hidden_layers': args.n_hidden_layers,
    'n_hidden_units': args.n_hidden_units,
    'n_samples': 1,
    'decay_rate': .8,
    'decay_steps': 5000,
    'starting_learning_rate': .01,
    'n_steps_enc': 1,
    'custom_lr': False,
    'use_dropout': True,
    'use_adamw': False,
    'alpha': .01,
    'scale_type': 'mean',
    'tot_epochs': 2,
    'batch_size': 200,
    'seed': 1,
    'gpu_mem': .9,
}

model_config_single = model_config.copy()
model_config_single.update({'model_name': 'lda_orig', 'architecture': 'standard', 'n_hidden_layers': 2})
model_config_single.update({'starting_learning_rate': .01, 'tot_epochs': 4, 'batch_size': 150, 'decay_steps': 800})

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
    train_betas = np.ones((n_topics, vocab_size)) * train_alpha
    valid_betas = np.ones((n_topics, vocab_size)) * valid_alpha
    train_topics = generate_topics(n=num_train_topics, betas=train_betas, seed=0, shuffle=True)
    valid_topics = generate_topics(n=num_valid_topics, betas=valid_betas, seed=1, shuffle=True)
    if args.mdreviews:
        test_topics = np.load('datasets/mdreviews/test_topics_3k_new.npy')
    else:
        test_betas = []
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
            test_betas.append(beta)
            test_topics = generate_topics(n=num_test_topics, betas=test_betas, seed=2, shuffle=True)

    if not args.mdreviews:
        for i, topics in zip(range(10), train_topics):
            plot_side_by_side_docs(topics, os.path.join(results_dir, 'train_topics_{}.pdf'.format(str(i).zfill(3))))
        for i, topics in enumerate(valid_topics):
            plot_side_by_side_docs(topics, os.path.join(results_dir, 'valid_topics_{}.pdf'.format(str(i).zfill(3))))
        for i, topics in enumerate(test_topics):
            plot_side_by_side_docs(topics, os.path.join(results_dir, 'test_topics_{}.pdf'.format(str(i).zfill(3))))

    if args.mdreviews:
        documents = np.load('datasets/mdreviews/train_3k_vocab.npy')
    else:
        documents, doc_topic_dists = generate_documents(test_topics[0], num_documents, n_topics, vocab_size, avg_num_words, alpha=.01, seed=0)
    if not args.mdreviews:
        plot_side_by_side_docs(documents[:40], os.path.join(results_dir, 'documents.pdf'))

    np.save(os.path.join(results_dir, 'train_topics.npy'), train_topics)
    np.save(os.path.join(results_dir, 'valid_topics.npy'), valid_topics)
    np.save(os.path.join(results_dir, 'test_topics.npy'), test_topics)
    np.save(os.path.join(results_dir, 'documents.npy'), documents)
    if not args.mdreviews:
        np.save(os.path.join(results_dir, 'document_topic_dists.npy'), doc_topic_dists)

logging.info('Data acquired')
get_memory_consumption()

def generate_datasets(train, valid, test, n):
    datasets = [train, valid, test]
    for dataset in datasets:
        docs, topics = dataset
        subset = [comb for _, comb in zip(range(n), itertools.product(docs, topics))]
        yield subset


if args.evaluate or args.evaluate_svi_convergence or args.evaluate_svi_convergence_with_vae_init:
    train = (documents, train_topics)
    valid = (documents, valid_topics)
    test = (documents, test_topics)

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
        if torch_isnan(loss):
            break
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
    svi = SVI(vae.model, vae.mean_field_guide, pyro_scheduler, loss=Trace_ELBO(), num_samples=100)
    z_loc, z_scale = vae.encoder.forward(data, topics)
    print(z_loc[0])
    print(z_scale[0])
    pyro.clear_param_store()
    pyro.get_param_store().get_param('z_loc', init_tensor=z_loc.detach())
    pyro.get_param_store().get_param('z_scale', init_tensor=z_scale.detach())
    print(pyro.get_param_store().get_all_param_names())
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
    train = (documents, train_topics)
    valid = (documents[:2], valid_topics)
    logging.info('Starting train')
    train_save_VAE(
        train, valid, model_config,
        training_epochs=model_config['tot_epochs'], batch_size=model_config['batch_size'],
        hallucinations=False, tensorboard=True, shuffle=True, display_step=100,
        n_topics=n_topics, vocab_size=vocab_size, recreate_docs=False, save_iter=1000, generate_train=True)
    logging.info('Finished train')
if args.train_single:
    logging.info('Starting training single')
    single_train = (documents, np.array([train_topics[random_topics_idx]]))
    valid = (documents[:2], valid_topics)
    train_save_VAE(
        single_train, valid, model_config_single,
        training_epochs=model_config_single['tot_epochs'], batch_size=model_config_single['batch_size'],
        hallucinations=False, tensorboard=True, shuffle=True, display_step=100,
        n_topics=n_topics, vocab_size=vocab_size, recreate_docs=False, save_iter=1000, generate_train=True)
    logging.info('Finished training single')
# load the VAE into pyro for evaluation
if args.evaluate:
    vae = VAE_pyro(**model_config)
    if args.new_model_path:
        vae.load_state_dict(torch.load(os.path.join(results_dir, 'ape.dict')), map_location="cpu")
    else:
        state_dict = vae.load()
        vae.load_state_dict(state_dict)

    if model_config['scale_type'] == 'mean':
        print(vae.encoder.scale)
    elif model_config['scale_type'] == 'sample':
        print(vae.decoder.scale)

    for data_name, data_and_topics in zip(dataset_names, datasets):
        data, topics = unzip_X_and_topics(data_and_topics)
        # save the total number of words in the data we are looking at
        with open(os.path.join(results_dir, 'num_words.csv'), 'a') as f:
            csv_writer = csv.writer(f)
            num_words = data.sum()
            csv_writer.writerow([data_name, num_words])
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

        if args.run_standard_vae:
            logging.info('Starting VAE single evaluation')
            pyro.clear_param_store()
            vae_single = VAE_pyro(**model_config_single)
            state_dict = vae_single.load()
            vae_single.load_state_dict(state_dict)
            single_vae_posterior_eval = partial(
                run_posterior_evaluation,
                data=data, data_name=data_name, topics=topics, vae=vae_single, sample_idx=sample_idx, model_config=model_config_single)
            vae_svi_single = SVI(vae_single.model, vae_single.encoder_guide, pyro_scheduler, loss=Trace_ELBO(), num_steps=0, num_samples=100)
            vae_svi_single = single_vae_posterior_eval(vae_svi_single, 'vae_single')
            get_memory_consumption()
            logging.info('Deleting vae_svi_single')
            del vae_svi_single
            get_memory_consumption()

        if args.run_svi:
            if args.run_standard_vae:
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
            
            logging.info('Starting SVI under time constraint')
            start = time.time()
            num_steps_to_try = [3, 4, 5]
            lrs = [.1, .2, .5]
            svi_stats = []
            for num_steps in num_steps_to_try:
                for lr in lrs:
                    pyro.clear_param_store()
                    start = time.time()
                    pyro_scheduler = StepLR({'optimizer': torch.optim.Adam, 'optim_args': {"lr": lr}, 'step_size': 10000, 'gamma': 0.95})
                    svi = SVI(vae.model, vae.mean_field_guide, pyro_scheduler, loss=Trace_ELBO(), num_samples=100)
                    for _ in range(num_steps):
                        loss = -svi.step(data, topics)
                    end = time.time()
                    posterior = svi.run(data, topics)
                    posterior_predictive = TracePredictive(vae.model, posterior, num_samples=10)
                    posterior_predictive_traces = posterior_predictive.run(data, topics)
                    # get the posterior predictive log likelihood
                    posterior_predictive_density = evaluate_log_predictive_density(posterior_predictive_traces)
                    posterior_predictive_density = float(posterior_predictive_density.detach().numpy())
                    svi_stats.append([data_name, num_steps, lr, end - start, posterior_predictive_density])
                    del svi
            get_memory_consumption()
            with open(os.path.join(results_dir, 'short_svi.csv'), 'a') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['data', 'num_steps', 'lr', 'time', 'posterior_predictive'])
                for row in svi_stats:
                    csv_writer.writerow(row)

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
    if not args.mdreviews:
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
    
    if args.plot:
        logging.info('Plotting SVI vs VAE ELBOs')
        pyro.clear_param_store()
        # reload vae to be able to take in different-sized batches
        vae = VAE_pyro(**model_config)
        state_dict = vae.load()
        vae.load_state_dict(state_dict)
        get_elbo_csv(vae, vae_single, results_dir)
        plot_svi_vs_vae_elbo_v1(results_dir)
        get_elbo_csv(vae, vae_single, results_dir, posterior_predictive=True)
        plot_svi_vs_vae_elbo_v1(results_dir, posterior_predictive=True)

        plot_times(results_dir)