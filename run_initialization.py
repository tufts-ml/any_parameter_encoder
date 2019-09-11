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
from models.lda_lognorm import VAE_tf, VAE_pyro
from training.train_vae import train
from common import train_save_VAE, save_elbo_vs_m, save_loglik_to_csv, save_reconstruction_array
from visualization.reconstructions import plot_side_by_side_docs, plot_saved_samples
from visualization.posterior import plot_posterior
from utils import softmax, unzip_X_and_topics

# where to write the results
results_file = 'results.csv'
results_dir = 'experiments/vae_experiments/maml_init_naive'
maml_init = True
print(results_dir)

# global params
n_topics = 18
# n_topics = 22
num_examples = 10
sample_idx = list(range(10))

# toy bars data
vocab_size = 100
train_topics = [permuted_toy_bars(m, m) for m in range(30, 300, 30)]
train_m = list(range(30, 300, 30))
train_documents, train_doc_topic_dists = generate_documents(toy_bars(), 10000, alpha=.01, seed=0)
valid_documents, valid_doc_topic_dists = generate_documents(toy_bars(), 100, alpha=.01, seed=1)
# train_documents, train_doc_topic_dists = generate_documents(toy_bars(), 100, alpha=.01)
# valid_documents, valid_doc_topic_dists = generate_documents(toy_bars(), 100, alpha=.01)

valid_topics = [toy_bars()]
valid_m = [0]
# TODO: perform correct queuing so full dataset doesn't need to be in memory
train_data_meta = list(itertools.product(train_documents, train_topics))
valid_data_meta = list(itertools.product(train_documents, valid_topics))

train_data = valid_data_meta
valid_data = list(itertools.product(valid_documents, valid_topics))

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
    'decay_rate': .5,
    'decay_steps': 1000,
    'starting_learning_rate': .01,
    'n_steps_enc': 1
}

# if not os.path.exists(results_dir):
#     os.system('mkdir -p ' + results_dir)
# shutil.copy(os.path.abspath(__file__), os.path.join(results_dir, 'run_simple.py'))

# get a good initialization with MAML
if maml_init:
    vae = train_save_VAE(train_data_meta, valid_data_meta, model_config, training_epochs=100, batch_size=200, hallucinations=False, tensorboard=True)
    vars_to_load = [v for v in tf.global_variables() if 'generator_network' not in v.name]
    saver = tf.train.Saver(var_list = vars_to_load)
    save_path = saver.save(vae.sess, os.path.join(results_dir, "model.ckpt"))
    vae.sess.close()
    tf.reset_default_graph()
    # run training on typical vae
    new_saver = tf.train.Saver(var_list = vars_to_load)

model_config['model_name'] = 'lda_orig'
vae_orig = VAE_tf(tensorboard=True, **model_config)
if maml_init:
    new_saver = tf.train.import_meta_graph(os.path.join(results_dir, "model.ckpt.meta"))
    save_path = new_saver.restore(vae_orig.sess, os.path.join(results_dir, "model.ckpt"))
tensorboard_logs_dir = os.path.join(
        model_config['results_dir'], model_config['model_name'],
        'logs_{}_{}'.format(model_config['n_hidden_layers'], model_config['n_hidden_units']))
vae_orig = train(train_data, valid_data, vae_orig, training_epochs=80, tensorboard=True, batch_size=200,
            tensorboard_logs_dir=tensorboard_logs_dir, results_dir=model_config['results_dir'], vae_meta=False)
vae_orig.save()
# save_elbo_vs_m(vae, all_documents, all_topics, all_m, results_dir)
final_topics = softmax(vae_orig.topic_prop(train_data[:1]))
print(final_topics.shape)

# load the VAE into pyro for evaluation
vae = VAE_pyro(**model_config)
state_dict = vae.load()
vae.load_state_dict(state_dict)
# for data_name, data_and_topics in zip(dataset_names, datasets):
for data_name, data_and_topics in zip(['train', 'valid'], [train_data[:100], valid_data[:100]]):
    data, topics = unzip_X_and_topics(data_and_topics)
    if model_config['architecture'] == 'naive':
        topics = np.array([final_topics] * len(data))
    data = torch.from_numpy(data.astype(np.float32))
    topics = torch.from_numpy(topics.astype(np.float32))
    pyro_scheduler = StepLR({'optimizer': torch.optim.Adam, 'optim_args': {"lr": .1}, 'step_size': 10000, 'gamma': 0.95})
    # pyro scheduler doesn't have any effect in the VAE case since we never take any optimization steps
    vae_svi = SVI(vae.model, vae.encoder_guide, pyro_scheduler, loss=Trace_ELBO(), num_steps=100, num_samples=100)
    svi = SVI(vae.model, vae.mean_field_guide, pyro_scheduler, loss=TraceMeanField_ELBO(), num_steps=100, num_samples=100)
    # mcmc = MCMC(NUTS(vae.model, adapt_step_size=True), num_samples=100, warmup_steps=50)
    for inference_name, inference in zip(['vae', 'svi'], [vae_svi, svi]):
    # for inference_name, inference in zip(['mcmc'], [mcmc]):
        print(inference_name)
    # for inference_name, inference in zip(['svi'], [svi]):
        # try:
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
        # except Exception as e:
        #     print(e)
        #     print(data_name, inference_name, " failed")
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
# for data_name, data in zip(dataset_names, [train_documents, valid_documents, test_documents]):
for data_name, data in zip(['train', 'valid'], [train_documents[:100], valid_documents[:100]]):
    filenames = []
    for inference in ['vae', 'svi']:
        file = '_'.join([inference, model_config['model_name'], data_name, str(model_config['n_hidden_layers']), str(model_config['n_hidden_units'])]) + '.npy'
        filepath = os.path.join(os.getcwd(), results_dir, file)
        if os.path.exists(filepath):
            filenames.append(filepath)

    plot_name = os.path.join(results_dir, data_name + '_vae_reconstructions.pdf')
    plot_saved_samples(np.array(data)[sample_idx], filenames, plot_name, vocab_size=vocab_size, intensity=10)

for i in range(10):
    plot_posterior(results_dir, i, ['train', 'valid'], ['vae', 'svi'])