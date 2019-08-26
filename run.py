import os
import shutil
import itertools
import numpy as np
import torch

from pyro.optim import StepLR
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from pyro.infer.mcmc import MCMC, NUTS
import pyro
import tensorflow as tf

from data.topics import toy_bars, permuted_toy_bars, diagonal_bars
from data.documents import generate_documents
from models.lda_meta import VAE_pyro
from common import train_save_VAE, save_elbo_vs_m, save_loglik_to_csv, save_reconstruction_array
from visualization.reconstructions import plot_side_by_side_docs, plot_saved_samples
from visualization.posterior import plot_posterior
from utils import softmax, unzip_X_and_topics

# where to write the results
results_dir = 'experiments/vae_experiments/naive_reconstructions5'
results_file = 'results.csv'
print(results_dir)

# global params
n_topics = 18
# n_topics = 22
num_examples = 10
sample_idx = list(range(10))

# toy bars data
vocab_size = 100
train_topics = [toy_bars()] + [permuted_toy_bars(m, m) for m in range(1, 91, 10)]
valid_topics = [permuted_toy_bars(50, 50)]
# test_topics = [diagonal_bars()]
train_m = [0] + list(range(1, 101, 10))
valid_m = [50]
# test_m = [162]
train_documents = []
train = []
for topics, m in zip(train_topics, train_m):
    plot_side_by_side_docs(topics, name="train_topics_{}.png".format(m))
    docs, _ = generate_documents(topics, 10, alpha=.01)
    train_documents.extend(docs)
    train.extend([(d, topics) for d in docs])

valid_documents, valid_doc_topic_dists = generate_documents(valid_topics[0], 100, alpha=.01)
# test_documents, test_doc_topic_dists = generate_documents(test_topics[0], 100, alpha=.01)
# TODO: perform correct queuing so full dataset doesn't need to be in memory
valid = [(d, valid_topics[0]) for d in valid_documents]
# test = [(d, test_topics[0]) for d in test_documents]
all_documents = [train_documents[:10], valid_documents]
all_topics = [train_topics, valid_topics]
all_m = [train_m, valid_m]
datasets = [train[:100], valid]
dataset_names = ['train', 'valid']

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
    'decay_steps': 50,
    'starting_learning_rate': .01,
    'n_steps_enc': 1
}

if not os.path.exists(results_dir):
    os.system('mkdir -p ' + results_dir)
shutil.copy(os.path.abspath(__file__), os.path.join(results_dir, 'run_simple.py'))

# train the VAE and save the weights
# vae = train_save_VAE(train, valid, model_config, training_epochs=150, batch_size=200, hallucinations=False, tensorboard=True)
# save_elbo_vs_m(vae, all_documents, all_topics, all_m, results_dir)
# vae.save()
# vae.sess.close()
# tf.reset_default_graph()
# load the VAE into pyro for evaluation
vae = VAE_pyro(**model_config)
state_dict = vae.load()
vae.load_state_dict(state_dict)
for data_name, data_and_topics in zip(dataset_names, datasets):
    data, topics = unzip_X_and_topics(data_and_topics)
    data = torch.from_numpy(data.astype(np.float32))
    topics = torch.from_numpy(topics.astype(np.float32))
    pyro_scheduler = StepLR({'optimizer': torch.optim.Adam, 'optim_args': {"lr": .1}, 'step_size': 10000, 'gamma': 0.95})
    # pyro scheduler doesn't have any effect in the VAE case since we never take any optimization steps
    # vae_svi = SVI(vae.model, vae.encoder_guide, pyro_scheduler, loss=Trace_ELBO(), num_steps=100, num_samples=100)
    # svi = SVI(vae.model, vae.mean_field_guide, pyro_scheduler, loss=TraceMeanField_ELBO(), num_steps=100, num_samples=100)
    mcmc = MCMC(NUTS(vae.model, adapt_step_size=True), num_samples=100, warmup_steps=50)
    for inference_name, inference in zip(['mcmc'], [mcmc]):
        print(inference_name)
        posterior = inference.run(data, topics)
        model_config.update({
            'data_name': data_name,
            'inference': inference_name,
            'results_dir': results_dir,
        })
        print(model_config)
        save_reconstruction_array(vae, topics, posterior, sample_idx, model_config)
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

for data_name, data in zip(dataset_names, all_documents):
    filenames = []
    for inference in ['vae', 'svi']:
        file = '_'.join([inference, model_config['model_name'], data_name, str(model_config['n_hidden_layers']), str(model_config['n_hidden_units'])]) + '.npy'
        filepath = os.path.join(os.getcwd(), results_dir, file)
        if os.path.exists(filepath):
            filenames.append(filepath)

    plot_name = os.path.join(results_dir, data_name + '_vae_reconstructions.pdf')
    plot_saved_samples(np.array(data)[sample_idx], filenames, plot_name, vocab_size=vocab_size, intensity=10)

for i in range(10):
    plot_posterior(results_dir, i, ['train', 'valid'], ['vae', 'svi', 'mcmc'])