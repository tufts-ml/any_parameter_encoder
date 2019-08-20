import os
import csv
import numpy as np
import tensorflow as tf
import itertools

from pyro.infer.abstract_infer import TracePredictive
from evaluation.evaluate_posterior import evaluate_log_predictive_density
from evaluation.evaluate_posterior import reconstruct_data
from visualization.reconstructions import plot_side_by_side_docs

from models.lda_meta import VAE_tf, VAE_pyro
from training.train_vae import train, train_with_hallucinations


def train_save_VAE(train_data, valid_data, model_config, training_epochs=120, batch_size=200, tensorboard=True, hallucinations=False):
    vae = VAE_tf(tensorboard=tensorboard, **model_config)
    tensorboard_logs_dir = os.path.join(
        model_config['results_dir'], model_config['model_name'],
        'logs_{}_{}'.format(model_config['n_hidden_layers'], model_config['n_hidden_units']))
    if hallucinations:
        vae = train_with_hallucinations(train_data, valid_data, vae, model_config, training_epochs=training_epochs, batch_size=batch_size,
            tensorboard=tensorboard, tensorboard_logs_dir=tensorboard_logs_dir, results_dir=model_config['results_dir'])
    else:
        vae = train(train_data, valid_data, vae, training_epochs=training_epochs, tensorboard=tensorboard, batch_size=batch_size,
            tensorboard_logs_dir=tensorboard_logs_dir, results_dir=model_config['results_dir'])
    return vae


def save_elbo_vs_m(vae, documents, topics, ms, results_dir):
    train_documents, valid_documents, test_documents = documents
    train_topics, valid_topics, test_topics = topics
    with open(os.path.join(results_dir, 'elbo_vs_m.csv'), 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['docs', 'topics', 'elbo', 'm'])
        for doc_set, data in zip(['train', 'valid', 'test'], documents):
            for topic_set, topics_set, m in zip(['train', 'valid', 'test'], topics, ms):
                for topic in topics_set:
                    elbo = vae.evaluate(list(itertools.product(data, [topic])))
                    writer.writerow([doc_set, topic_set, elbo, m])



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

