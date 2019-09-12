import os
import csv
import numpy as np
import tensorflow as tf
import itertools

from pyro.infer.abstract_infer import TracePredictive
from pyro.infer.util import torch_item
from pyro.optim import StepLR
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from pyro.infer.mcmc import MCMC, NUTS
from pyro.util import torch_isnan
import pyro
import tensorflow as tf
import torch
from evaluation.evaluate_posterior import evaluate_log_predictive_density
from evaluation.evaluate_posterior import reconstruct_data
from visualization.reconstructions import plot_side_by_side_docs

from models.lda_meta import VAE_tf, VAE_pyro
from training.train_vae import train, train_with_hallucinations
from utils import unzip_X_and_topics


def train_save_VAE(train_data, valid_data, model_config, training_epochs=120, batch_size=200, tensorboard=True, hallucinations=False, shuffle=True, display_step=5):
    vae = VAE_tf(tensorboard=tensorboard, **model_config)
    tensorboard_logs_dir = os.path.join(
        model_config['results_dir'], model_config['model_name'],
        'logs_{}_{}'.format(model_config['n_hidden_layers'], model_config['n_hidden_units']))
    if hallucinations:
        vae = train_with_hallucinations(train_data, valid_data, vae, model_config, training_epochs=training_epochs, batch_size=batch_size,
            tensorboard=tensorboard, tensorboard_logs_dir=tensorboard_logs_dir, results_dir=model_config['results_dir'])
    else:
        vae = train(train_data, valid_data, vae, training_epochs=training_epochs, tensorboard=tensorboard, batch_size=batch_size,
            tensorboard_logs_dir=tensorboard_logs_dir, results_dir=model_config['results_dir'], display_step=display_step)
    vae.save()
    vae.sess.close()
    tf.reset_default_graph()


def save_speed_to_csv(model_config, clock_time):
    with open(os.path.join(model_config['results_dir'], 'clock_times.csv'), 'a') as f:
        row = [model_config['data_name'], model_config['inference'], clock_time]
        csv_writer = csv.writer(f)
        csv_writer.writerow(row)


def save_elbo_vs_m(vae, documents, topics, ms, results_dir, names=['train', 'valid']):
    with open(os.path.join(results_dir, 'elbo_vs_m.csv'), 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['docs', 'topics', 'elbo', 'm'])
        for doc_set, data in zip(names, documents):
            for topic_set, topics_set, m_set in zip(names, topics, ms):
                for topic, m in zip(topics_set, m_set):
                    elbo = vae.evaluate(list(itertools.product(data, [topic])))
                    writer.writerow([doc_set, topic_set, elbo, m])


def get_elbo_vs_m(vae, dataset_names, datasets, results_dir, distances):
    with open(os.path.join(results_dir, 'elbo_vs_m.csv'), 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['data', 'inference', 'elbo', 'm'])
        for data_name, data_and_topics in zip(dataset_names, datasets):
            topics, data = unzip_X_and_topics(data_and_topics)
            data = torch.from_numpy(data.astype(np.float32))
            topics = torch.from_numpy(topics.astype(np.float32))
            pyro_scheduler = StepLR({'optimizer': torch.optim.Adam, 'optim_args': {"lr": .1}, 'step_size': 10000, 'gamma': 0.95})
            for i, m in enumerate(distances):
                data_i = data[i * 100: (i + 1) * 100]
                topics_i = topics[i * 100: (i + 1) * 100]

                vae_elbo = Trace_ELBO()
                vae_svi = SVI(vae.model, vae.encoder_guide, pyro_scheduler, loss=vae_elbo, num_steps=300, num_samples=100)

                vae_posterior = vae_svi.run(data_i, topics_i)
                writer.writerow([data_name, 'vae', vae_svi.evaluate_loss(data_i, topics_i), m])
                svi_loss = np.nan
                while torch_isnan(svi_loss):
                    svi_elbo = Trace_ELBO()
                    svi = SVI(vae.model, vae.mean_field_guide, pyro_scheduler, loss=svi_elbo, num_steps=400, num_samples=100)
                    svi_posterior = svi.run(data_i, topics_i)
                    svi_loss = svi.evaluate_loss(data_i, topics_i)
                    pyro.clear_param_store()
                writer.writerow([data_name, 'svi', svi_loss, m])
                pyro.clear_param_store()



def save_loglik_to_csv(data, topics, model, posterior, model_config, num_samples=10):
    posterior_predictive = TracePredictive(model, posterior, num_samples=num_samples)
    posterior_predictive_traces = posterior_predictive.run(data, topics)
    # get the posterior predictive log likelihood
    posterior_predictive_density = evaluate_log_predictive_density(posterior_predictive_traces)
    posterior_predictive_density = float(posterior_predictive_density.detach().numpy())
    results_dir = model_config['results_dir']
    inference = model_config['inference']
    model_name = model_config['model_name']
    data_name = model_config['data_name']
    n_hidden_layers = model_config['n_hidden_layers']
    n_hidden_units = model_config['n_hidden_units']
    results_file = model_config['results_file']
    with open(os.path.join(results_dir, results_file), 'a') as f:
        row = [inference, model_name, data_name, n_hidden_layers, n_hidden_units, posterior_predictive_density]
        csv_writer = csv.writer(f)
        csv_writer.writerow(row)


def save_reconstruction_array(vae, topics, posterior, sample_idx, model_config):
    # reconstruct the data
    reconstructions = reconstruct_data(posterior, vae, topics)
    # save sample reconstructions
    averaged_reconstructions = np.mean(reconstructions[:, sample_idx], axis=0)
    results_dir = model_config['results_dir']
    inference = model_config['inference']
    model_name = model_config['model_name']
    data_name = model_config['data_name']
    n_hidden_layers = model_config['n_hidden_layers']
    n_hidden_units = model_config['n_hidden_units']
    file = os.path.join(
        results_dir, '_'.join([inference, model_name, data_name, str(n_hidden_layers), str(n_hidden_units)])) + '.npy'
    np.save(file, averaged_reconstructions)

