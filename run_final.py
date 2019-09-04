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

from data.topics import toy_bars, permuted_toy_bars, diagonal_bars, get_random_topics
from data.documents import generate_documents
from models.lda_meta import VAE_pyro
from common import train_save_VAE, get_elbo_vs_m, save_loglik_to_csv, save_reconstruction_array
from visualization.reconstructions import plot_side_by_side_docs, plot_saved_samples
from visualization.posterior import plot_posterior_dense
from utils import softmax, unzip_X_and_topics, normalize1d

# where to write the results
results_dir = 'experiments/vae_experiments/figures3'
results_file = 'results.csv'
print(results_dir)
if not os.path.exists(results_dir):
    os.system('mkdir -p ' + results_dir)
shutil.copy(os.path.abspath(__file__), os.path.join(results_dir, 'run.py'))

# global params
vocab_size = 100
n_topics = 18
num_examples = 10
sample_idx = list(range(10))

model_config = {
    'vocab_size': vocab_size,
    'n_topics': n_topics,
    'results_dir': results_dir,
    'results_file': results_file,
    'inference': 'vae',
    'model_name': 'lda_meta',
    'architecture': 'naive',
    'scale_trainable': True,
    'n_hidden_layers': 5,
    'n_hidden_units': 100,
    'n_samples': 1,
    'decay_rate': .9,
    'decay_steps': 1000,
    'starting_learning_rate': .01,
    'n_steps_enc': 1
}

# toy bars data
train_documents, train_doc_topic_dists = generate_documents(toy_bars(), 500, alpha=.01, seed=0)
valid_documents, valid_doc_topic_dists = generate_documents(toy_bars(), 100, alpha=.01, seed=1)

# topics
# train_topics = get_random_topics(n=20, num_topics=n_topics, vocab_size=vocab_size, alpha=.01, seed=0)
toy_bar_topics = toy_bars(normalized=False)
topics = np.copy(toy_bar_topics)
train_topics = []
for i in range(20):
    print(i)
    topics = permuted_toy_bars(topics, 60, seed=i, normalized=False)
    train_topics.append(map(normalize1d, topics))

for i, topics in enumerate(train_topics):
    plot_side_by_side_docs(np.array(topics), os.path.join(results_dir, 'train_topics_{}.pdf'.format(str(i).zfill(3))))
valid_topics = [toy_bars()]
plot_side_by_side_docs(np.array(valid_topics[0]), os.path.join(results_dir, 'true_topics.pdf'))
# TODO: perform correct queuing so full dataset doesn't need to be in memory
train = list(itertools.product(train_documents, train_topics))
valid = list(itertools.product(train_documents, valid_topics))

# train the VAE and save the weights
vae = train_save_VAE(train, valid, model_config, training_epochs=200, batch_size=200, hallucinations=False, tensorboard=True)

# topics for eval: true, diagonal, various perturbations
toy_bar_topics = toy_bars(normalized=False)
valid_topics = [map(normalize1d, toy_bar_topics), diagonal_bars()]

test_topics = [map(normalize1d, toy_bar_topics)]
topics = np.copy(toy_bar_topics)
for i in range(10):
    print(i)
    topics = permuted_toy_bars(topics, 30, seed=i + 1, normalized=False)
    test_topics.append(map(normalize1d, topics))
# test_topics = np.load(os.path.join(results_dir, 'test_topics.npy'))

# save data
np.save(os.path.join(results_dir, 'train_documents.npy'), train_documents)
np.save(os.path.join(results_dir, 'valid_documents.npy'), valid_documents)
np.save(os.path.join(results_dir, 'train_topics.npy'), train_topics)
np.save(os.path.join(results_dir, 'valid_topics.npy'), valid_topics)
np.save(os.path.join(results_dir, 'test_topics.npy'), test_topics)

# evaluation: posterior, reconstructions, posterior predictive likelihood, ELBO
valid_seen = list(itertools.product(train_documents[:100], valid_topics))
valid_unseen = list(itertools.product(valid_documents, valid_topics))

test_seen = list(itertools.product(test_topics, train_documents[:100]))
test_unseen = list(itertools.product(test_topics, valid_documents))
distances = list(range(30, 301, 30))

vae = VAE_pyro(**model_config)
state_dict = vae.load()
vae.load_state_dict(state_dict)
for data_name, data_and_topics in zip(['seen_docs', 'unseen_docs'], [valid_seen, valid_unseen]):
    data, topics = unzip_X_and_topics(data_and_topics)
    data = torch.from_numpy(data.astype(np.float32))
    topics = torch.from_numpy(topics.astype(np.float32))
    pyro_scheduler = StepLR({'optimizer': torch.optim.Adam, 'optim_args': {"lr": .1}, 'step_size': 10000, 'gamma': 0.95})
    # pyro scheduler doesn't have any effect in the VAE case since we never take any optimization steps
    vae_svi = SVI(vae.model, vae.encoder_guide, pyro_scheduler, loss=Trace_ELBO(), num_steps=1, num_samples=100)
    svi = SVI(vae.model, vae.mean_field_guide, pyro_scheduler, loss=Trace_ELBO(), num_steps=500, num_samples=100)
    mcmc = MCMC(NUTS(vae.model, adapt_step_size=True), num_samples=300, warmup_steps=50)
    for inference_name, inference in zip(['vae', 'svi', 'mcmc'], [vae_svi, svi, mcmc]):
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
        pyro.clear_param_store()

for data_name, data_and_topics in zip(['seen_docs', 'unseen_docs'], [valid_seen, valid_unseen]):
    data, _ = unzip_X_and_topics(data_and_topics)
    filenames = []
    for inference in ['vae', 'svi', 'mcmc']:
        file = '_'.join([inference, model_config['model_name'], data_name, str(model_config['n_hidden_layers']), str(model_config['n_hidden_units'])]) + '.npy'
        filepath = os.path.join(os.getcwd(), results_dir, file)
        if os.path.exists(filepath):
            filenames.append(filepath)

    plot_name = os.path.join(results_dir, data_name + '_reconstructions.pdf')
    plot_saved_samples(np.array(data)[sample_idx], filenames, plot_name, vocab_size=vocab_size, intensity=10)

for data_name in ['seen_docs', 'unseen_docs']:
    plot_posterior_dense(results_dir, sample_idx, data_name, ['vae', 'svi', 'mcmc'])

get_elbo_vs_m(vae, ['seen_docs', 'unseen_docs'], [test_seen, test_unseen], results_dir, distances)