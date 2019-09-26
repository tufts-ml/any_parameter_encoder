import os
import csv
import numpy as np
import tensorflow as tf
import itertools
import time

from pyro.infer.abstract_infer import TracePredictive
from pyro.infer.util import torch_item
from pyro.optim import StepLR
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from pyro.infer.mcmc import MCMC, NUTS
from pyro.util import torch_isnan
import pyro
import tensorflow as tf
import torch
from scipy.stats import multivariate_normal, norm

from evaluation.evaluate_posterior import evaluate_log_predictive_density
from evaluation.evaluate_posterior import reconstruct_data
from visualization.reconstructions import plot_side_by_side_docs

from models.lda_meta import VAE_tf, VAE_pyro
from training.train_vae import train, train_with_hallucinations
from utils import unzip_X_and_topics, softmax


MIN_LOG_PROB = -9999999999

def train_save_VAE(train_data, valid_data, model_config, training_epochs=120, batch_size=200,
                   tensorboard=True, hallucinations=False, shuffle=True, display_step=5, recreate_docs=True,
                   vocab_size=100, n_topics=20):
    vae = VAE_tf(tensorboard=tensorboard, **model_config)
    tensorboard_logs_dir = os.path.join(
        model_config['results_dir'], model_config['model_name'],
        'logs_{}_{}'.format(model_config['n_hidden_layers'], model_config['n_hidden_units']))
    if hallucinations:
        vae = train_with_hallucinations(train_data, valid_data, vae, model_config, training_epochs=training_epochs, batch_size=batch_size,
            tensorboard=tensorboard, tensorboard_logs_dir=tensorboard_logs_dir, results_dir=model_config['results_dir'], vocab_size=vocab_size,
            n_topics=n_topics)
    else:
        vae = train(train_data, valid_data, vae, training_epochs=training_epochs, tensorboard=tensorboard, batch_size=batch_size,
            tensorboard_logs_dir=tensorboard_logs_dir, results_dir=model_config['results_dir'], display_step=display_step)
    if recreate_docs:
        for data_name, batch_xs in zip(['train', 'valid'], [train_data[:10], valid_data[:10]]):
            recreated_docs, _, _ = vae.recreate_input(batch_xs)
            X, topics = unzip_X_and_topics(batch_xs)
            plot_side_by_side_docs(np.concatenate([X, recreated_docs]), os.path.join(model_config['results_dir'], 'recreated_docs_{}.pdf'.format(str(data_name).zfill(2))))
    vae.save()
    vae.sess.close()
    tf.reset_default_graph()


def run_posterior_evaluation(inference, inference_name, data, data_name, topics, vae, sample_idx, model_config):
    print(inference_name)
    model_config.update({'inference': inference_name})
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
        np.save(os.path.join(model_config['results_dir'], '{}_{}_samples.npy'.format(data_name, inference_name)), np.array(samples))
        log_prob_sums = [t.log_prob_sum() for t in posterior.exec_traces]
        np.save(os.path.join(model_config['results_dir'], '{}_{}_log_prob_sums.npy'.format(data_name, inference_name)), np.array(log_prob_sums))
    else:
        if inference_name == 'vae':
            z_loc, z_scale = vae.encoder.forward(data, topics)
            z_loc = z_loc.data.numpy()
            z_scale = z_scale.data.numpy()
        elif inference_name == 'svi':
            z_loc = pyro.get_param_store().match('z_loc')['z_loc'].detach().numpy()
            z_scale = pyro.get_param_store().match('z_scale')['z_scale'].detach().numpy()
        np.save(os.path.join(model_config['results_dir'], '{}_{}_z_loc.npy'.format(data_name, inference_name)), z_loc)
        np.save(os.path.join(model_config['results_dir'], '{}_{}_z_scale.npy'.format(data_name, inference_name)), z_scale)
    
    return inference


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


def get_elbo_csv(vae, results_dir, restart=True):
    dataset_names = ['train', 'valid', 'test']
    try:
        train_topics = np.load(os.path.join(results_dir, 'train_topics.npy'))
        valid_topics = np.load(os.path.join(results_dir, 'valid_topics.npy'))
        test_topics = np.load(os.path.join(results_dir, 'test_topics.npy'))
        documents = np.load(os.path.join(results_dir, 'documents.npy'))
    except:
        train_topics = np.load(os.path.join('experiments', 'train_topics.npy'))
        valid_topics = np.load(os.path.join('experiments', 'valid_topics.npy'))
        test_topics = np.load(os.path.join('experiments', 'test_topics.npy'))
        documents = np.load(os.path.join('experiments', 'documents.npy'))
    num_topics_by_data = {}
    num_topics_by_data['train'] = len(train_topics)
    num_topics_by_data['valid'] = len(valid_topics)
    num_topics_by_data['test'] = len(test_topics)
    train = list(itertools.product(train_topics, documents))
    valid = list(itertools.product(valid_topics, documents))
    test = list(itertools.product(test_topics, documents))
    datasets = [train, valid, test]
    num_docs = len(documents)
    with open(os.path.join(results_dir, 'elbos.csv'), 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['Dataset', 'Topic index', 'SVI ELBO', 'VAE encoder ELBO'])
        for data_name, topics_and_data in zip(dataset_names, datasets):
            topics, data = unzip_X_and_topics(topics_and_data)
            data = torch.from_numpy(data.astype(np.float32))
            topics = torch.from_numpy(topics.astype(np.float32))
            pyro_scheduler = StepLR({'optimizer': torch.optim.Adam, 'optim_args': {"lr": .1}, 'step_size': 10000, 'gamma': 0.95})
            num_topics = num_topics_by_data[data_name]

            for i in range(num_topics):
                data_i = data[i * num_docs: (i + 1) * num_docs]
                topics_i = topics[i * num_docs: (i + 1) * num_docs]
                vae_elbo = Trace_ELBO()
                vae_svi = SVI(vae.model, vae.encoder_guide, pyro_scheduler, loss=vae_elbo, num_steps=0, num_samples=100)

                vae_posterior = vae_svi.run(data_i, topics_i)
                vae_loss = -vae_svi.evaluate_loss(data_i, topics_i)
                svi_loss = np.nan
                if restart:
                    while torch_isnan(svi_loss):
                        svi_elbo = Trace_ELBO()
                        svi = SVI(vae.model, vae.mean_field_guide, pyro_scheduler, loss=svi_elbo, num_steps=400, num_samples=100)
                        svi_posterior = svi.run(data_i, topics_i)
                        svi_loss = -svi.evaluate_loss(data_i, topics_i)
                        pyro.clear_param_store()
                else:
                    svi_elbo = Trace_ELBO()
                    svi = SVI(vae.model, vae.mean_field_guide, pyro_scheduler, loss=svi_elbo, num_steps=400, num_samples=100)
                    svi_posterior = svi.run(data_i, topics_i)
                    svi_loss = -svi.evaluate_loss(data_i, topics_i)
                    pyro.clear_param_store()
                writer.writerow([data_name, i, svi_loss, vae_loss])
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


def save_kl_to_csv(results_dir, data_name):
    """ Compare SVI and VAE encoder to MCMC """
    with open(os.path.join(results_dir, 'kl_div.csv'), 'a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['data', 'inference', 'kl'])
        mcmc_samples = np.load(os.path.join(results_dir, '{}_{}_samples.npy'.format(data_name, 'mcmc')))
        mcmc_log_prob_sums = np.load(os.path.join(results_dir, '{}_{}_log_prob_sums.npy'.format(data_name, 'mcmc')))
        for inference_name in ['vae', 'svi']:
            z_loc = np.load(os.path.join(results_dir, '{}_{}_z_loc.npy'.format(data_name, inference_name)))
            z_scale = np.load(os.path.join(results_dir, '{}_{}_z_scale.npy'.format(data_name, inference_name)))
            num_samples = 0
            total = 0
            for sample_unnorm, log_prob_sum in zip(mcmc_samples, mcmc_log_prob_sums):
                log_qx = log_prob_sum
                log_px = 0
                for i, datapoint in enumerate(sample_unnorm):
                    # print(datapoint)
                    # TODO: fix nans here
                    # print(len(datapoint))
                    # print(len(z_loc[i]))
                    # print(len(z_scale[i]))
                    print([(d, l, s) for d, l, s in zip(datapoint, z_loc[i], z_scale[i])])
                    pxs = [norm.ppf(d, loc=l, scale=s) for d, l, s in zip(datapoint, z_loc[i], z_scale[i])]
                    pxs = [0 if np.isnan(px) else px for px in pxs]
                    print(len(pxs))
                    print(pxs)
                    log_px = np.sum([np.log(px) if px != 0 else MIN_LOG_PROB for px in pxs])
                    # diag_scale = z_scale[i] ** 2
                    # if np.any(diag_scale == 0):
                    #     print('a scale value is zero for {}'.format(inference_name))
                    #     diag_scale = diag_scale + 1e-10
                    # px = multivariate_normal.pdf(datapoint, mean=z_loc[i], cov=np.diag(diag_scale))
                    # if px != 0:
                    #     log_px += np.log(px)
                    # else:
                    #     log_px += MIN_LOG_PROB
                total += log_qx - log_px
                num_samples += 1
            # this is the forward KL, which is zero-avoiding
            kl_qp = total / num_samples
            writer.writerow([data_name, inference_name, kl_qp])


def save_reconstruction_array(vae, topics, posterior, sample_idx, model_config, average_over='reconstructions'):
    if average_over == 'reconstructions':
        # reconstruct the data
        reconstructions = reconstruct_data(posterior, vae, topics)
        # save sample reconstructions
        averaged_reconstructions = np.mean(reconstructions[:, sample_idx], axis=0)
    elif average_over == 'samples':
        zs = []
        for tr in posterior.exec_traces:
            zs.append(tr.nodes['latent']['value'].data.numpy())
        zs = np.array(zs)
        z = torch.from_numpy(np.mean(zs[:, sample_idx], axis=0).astype(np.float32))
        averaged_reconstructions = vae.decoder(z, topics[:10]).detach().numpy()
    results_dir = model_config['results_dir']
    inference = model_config['inference']
    model_name = model_config['model_name']
    data_name = model_config['data_name']
    n_hidden_layers = model_config['n_hidden_layers']
    n_hidden_units = model_config['n_hidden_units']
    file = os.path.join(
        results_dir, '_'.join([inference, model_name, data_name, str(n_hidden_layers), str(n_hidden_units)])) + '.npy'
    np.save(file, averaged_reconstructions)


if __name__ == "__main__":
    save_kl_to_csv(None, 'train', {'scale_type': 'sample', 'results_dir': 'experiments/naive_sample2'})