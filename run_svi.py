import os
import shutil
import itertools
import numpy as np
import torch
import csv

from pyro.optim import StepLR
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from pyro.infer.mcmc import MCMC, NUTS
from pyro.util import torch_isnan
import pyro
import tensorflow as tf
import matplotlib.pyplot as plt

from data.topics import toy_bars, permuted_toy_bars, diagonal_bars
from data.documents import generate_documents
from models.lda_meta import VAE_pyro
from common import train_save_VAE, save_elbo_vs_m, save_loglik_to_csv, save_reconstruction_array
from visualization.reconstructions import plot_side_by_side_docs, plot_saved_samples
from visualization.posterior import plot_posterior
from utils import softmax, unzip_X_and_topics, normalize1d

# where to write the results
results_dir = 'experiments/vae_experiments/final3'
results_file = 'results.csv'
print(results_dir)
if not os.path.exists(results_dir):
    os.system('mkdir -p ' + results_dir)
shutil.copy(os.path.abspath(__file__), os.path.join(results_dir, 'run_simple.py'))

# global params
n_topics = 18
# n_topics = 22
num_examples = 10
sample_idx = list(range(10))

# toy bars data
vocab_size = 100
# train_topics = []
# toy_bar_topics = toy_bars(normalized=False)
# topics = np.array(toy_bar_topics)
# for i in range(10):
#     print(i)
#     topics = permuted_toy_bars(topics, 30, seed=i, normalized=False)
#     train_topics.append(map(normalize1d, topics))
# np.save(os.path.join(results_dir, 'train_topics.npy'), train_topics)

# valid_topics = [map(normalize1d, toy_bar_topics)]
# topics = toy_bar_topics
# for i in range(9):
#     print(i)
#     topics = permuted_toy_bars(topics, 30, seed=i + 1, normalized=False)
#     valid_topics.append(map(normalize1d, topics))
# np.save(os.path.join(results_dir, 'valid_topics.npy'), valid_topics)


train_topics = np.load(os.path.join(results_dir, 'train_topics.npy'))
valid_topics = np.load(os.path.join(results_dir, 'valid_topics.npy'))

# toy_bar_topics = toy_bars()
# train_topics = [permuted_toy_bars(toy_bar_topics, m, m) for m in range(30, 301, 30)]
# valid_topics = [toy_bar_topics] + [permuted_toy_bars(toy_bar_topics, m, m + 2) for m in range(30, 300, 30)]
# test_topics = [diagonal_bars()]


train_m = list(range(30, 301, 30))
valid_m = [0] + list(range(30, 300, 30))
test_m = [0] + list(range(30, 300, 30))
for m, topics in zip(train_m, train_topics):
    plot_side_by_side_docs(topics, os.path.join(results_dir, 'train_topics_{}.pdf'.format(str(m).zfill(3))))
for m, topics in zip(valid_m, valid_topics):
    plot_side_by_side_docs(topics, os.path.join(results_dir, 'valid_topics_{}.pdf'.format(str(m).zfill(3))))

# np.save(os.path.join(results_dir, 'train_topics.npy'), train_topics)
# np.save(os.path.join(results_dir, 'valid_topics.npy'), valid_topics)
# train_documents = []
# train = []
# for topics, m in zip(train_topics, train_m):
#     plot_side_by_side_docs(topics, name="train_topics_{}.png".format(m))
#     docs, _ = generate_documents(topics, 10, alpha=.01)
#     train_documents.extend(docs)
#     train.extend([(d, topics) for d in docs])
# train_documents, train_doc_topic_dists = generate_documents(toy_bars(), 1000, alpha=.01, seed=0)
# valid_documents, valid_doc_topic_dists = generate_documents(toy_bars(), 100, alpha=.01, seed=1)

# np.save(os.path.join(results_dir, 'train_documents.npy'), train_documents)
# np.save(os.path.join(results_dir, 'valid_documents.npy'), valid_documents)

train_documents = np.load(os.path.join(results_dir, 'train_documents.npy'))[:100]
valid_documents = np.load(os.path.join(results_dir, 'valid_documents.npy'))
# TODO: perform correct queuing so full dataset doesn't need to be in memory
# train = list(itertools.product(train_documents, train_topics))
# valid = list(itertools.product(train_documents, valid_topics))
# test = list(itertools.product(valid_documents, valid_topics))

train = list(itertools.product(train_topics, train_documents))
valid = list(itertools.product(train_topics, valid_documents))
test = list(itertools.product(valid_topics, valid_documents))

all_documents = [train_documents, valid_documents]
all_topics = [train_topics, valid_topics]
all_m = {
    'train': train_m,
    'valid': valid_m,
    'test': test_m
}
datasets = [train, valid, test]
dataset_names = ['train', 'valid', 'test']

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
    'n_samples': 1,
    'decay_rate': .9,
    'decay_steps': 500,
    'starting_learning_rate': .01,
    'n_steps_enc': 1
}

# train the VAE and save the weights
# vae = train_save_VAE(train, valid, model_config, training_epochs=150, batch_size=200, hallucinations=False, tensorboard=True)
# save_elbo_vs_m(vae, all_documents, all_topics, all_m, results_dir)
# vae.save()
# vae.sess.close()
# tf.reset_default_graph()
# load the VAE into pyro for evaluation
num_topic_sets = 10
vae = VAE_pyro(**model_config)
state_dict = vae.load()
vae.load_state_dict(state_dict)
with open(os.path.join(results_dir, 'elbo_vs_m1.csv'), 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['data', 'inference', 'elbo', 'm'])
    for data_name, data_and_topics in zip(dataset_names, datasets):
        topics, data = unzip_X_and_topics(data_and_topics)
        data = torch.from_numpy(data.astype(np.float32))
        topics = torch.from_numpy(topics.astype(np.float32))
        pyro_scheduler = StepLR({'optimizer': torch.optim.Adam, 'optim_args': {"lr": .1}, 'step_size': 10000, 'gamma': 0.95})
        # pyro scheduler doesn't have any effect in the VAE case since we never take any optimization steps
        # mcmc = MCMC(NUTS(vae.model, adapt_step_size=True), num_samples=100, warmup_steps=50)
        # for inference_name, inference in zip(['vae', 'svi', 'mcmc'], [vae_svi, svi, mcmc]):
            # print(inference_name)
        for i, m in enumerate(all_m[data_name]):
            vae_elbo = Trace_ELBO()
            svi_elbo = Trace_ELBO()
            vae_svi = SVI(vae.model, vae.encoder_guide, pyro_scheduler, loss=vae_elbo, num_steps=300, num_samples=100)
            data_i = data[i * 100: (i + 1) * 100]
            topics_i = topics[i * 100: (i + 1) * 100]
            vae_posterior = vae_svi.run(data_i, topics_i)
            writer.writerow([data_name, 'vae', vae_svi.evaluate_loss(data_i, topics_i), m])
            svi_loss = np.nan
            while torch_isnan(svi_loss):
                svi = SVI(vae.model, vae.mean_field_guide, pyro_scheduler, loss=svi_elbo, num_steps=400, num_samples=100)
                svi_posterior = svi.run(data_i, topics_i)
                svi_loss = svi.evaluate_loss(data_i, topics_i)
                pyro.clear_param_store()
            writer.writerow([data_name, 'svi', svi_loss, m])
            print('vae ', m, vae_svi.evaluate_loss(data_i, topics_i))
            print('svi ', m, svi_loss)
            pyro.clear_param_store()
        #     model_config.update({
        #         'data_name': data_name,
        #         'inference': inference_name,
        #         'results_dir': results_dir,
        #     })
        # print(model_config)
        # save_reconstruction_array(vae, topics, posterior, sample_idx, model_config, num_topic_sets)
#         for i in range(10):
#             # saves a separate row to the csv
#             save_loglik_to_csv(data, topics, vae.model, posterior, model_config, num_samples=10)
#         # save the posteriors for later analysis
#         if inference_name == 'mcmc':
#             # [num_samples, num_docs, num_topics]
#             samples = [t.nodes['latent']['value'].detach().cpu().numpy() for t in posterior.exec_traces]
#             np.save(os.path.join(results_dir, '{}_{}_samples.npy'.format(data_name, inference_name)), np.array(samples))
#         else:
#             if inference_name == 'vae':
#                 z_loc, z_scale = vae.encoder.forward(data, topics)
#                 z_loc = z_loc.data.numpy()
#                 z_scale = z_scale.data.numpy()
#             elif inference_name == 'svi':
#                 z_loc = pyro.get_param_store().match('z_loc')['z_loc'].detach().numpy()
#                 z_scale = pyro.get_param_store().match('z_scale')['z_scale'].detach().numpy()
#             np.save(os.path.join(results_dir, '{}_{}_z_loc.npy'.format(data_name, inference_name)), z_loc)
#             np.save(os.path.join(results_dir, '{}_{}_z_scale.npy'.format(data_name, inference_name)), z_scale)

# for data_name, data_and_topics in zip(dataset_names, datasets):
#     data, _ = unzip_X_and_topics(data_and_topics)
#     filenames = []
#     for inference in ['vae', 'svi', 'mcmc']:
#         file = '_'.join([inference, model_config['model_name'], data_name, str(model_config['n_hidden_layers']), str(model_config['n_hidden_units'])]) + '.npy'
#         filepath = os.path.join(os.getcwd(), results_dir, file)
#         if os.path.exists(filepath):
#             filenames.append(filepath)

#     plot_name = os.path.join(results_dir, data_name + '_vae_reconstructions.pdf')
#     plot_saved_samples(np.array(data)[sample_idx], filenames, plot_name, vocab_size=vocab_size, intensity=10)

# for i in sample_idx:
#     plot_posterior(results_dir, i, ['train', 'valid', 'test'], ['vae', 'svi', 'mcmc'])