import importlib
import numpy as np
import tensorflow as tf
import torch
from datasets.load import load_toy_bars
from training.train_vae import train

import concurrent.futures

# global params
topic_init_and_fixed = True
datadir = 'toy_bars'
vocab_size = 9
n_topics = 4

# vae params
n_runs = 5
vae_params = {
    'n_hidden_layers': [1, 2, 5],
    'n_hidden_units': [10, 20, 50, 100]
}
model = 'models.lda_lognorm'
mod = importlib.import_module(model)

# inference params
inference_techniques = ['vae', 'svi', 'hmc']


# 0. load data
data_tr, data_va, data_te = load_toy_bars(datadir, vocab_size)

def train_save_VAE(n_hidden_layers, n_hidden_units):
    vae = mod.VAE_tf(n_hidden_layers=n_hidden_layers, n_hidden_units=n_hidden_units, n_topics=n_topics,
                     vocab_size=vocab_size, tensorboard=True)
    vae = train(data_tr, vae, training_epochs=100, tensorboard=True,
                tensorboard_logs_dir='dump/logs_{}_{}'.format(n_hidden_layers, n_hidden_units))
    vae.save()
    vae.sess.close()
    tf.reset_default_graph()


# 1. train vae
with concurrent.futures.ProcessPoolExecutor() as executor:
    for n_hidden_layers in vae_params['n_hidden_layers']:
        for n_hidden_units in vae_params['n_hidden_units']:
            future = executor.submit(train_save_VAE, n_hidden_layers, n_hidden_units)
            print("Run with: %s | %s" % (n_hidden_layers, n_hidden_units))

    for future in concurrent.futures.as_completed():
            try:
                future.result()
            except Exception as exc:
                print('Exception raised: %s' % (exc))

# 2. infer from model
sample_docs = torch.from_numpy(data_tr[:5].astype(np.float32))
for n_hidden_layers in vae_params['n_hidden_layers']:
    for n_hidden_units in vae_params['n_hidden_units']:
        vae = mod.VAE_pyro(n_hidden_layers=n_hidden_layers, n_hidden_units=n_hidden_units, n_topics=n_topics, vocab_size=vocab_size)
        state_dict = vae.load()
        vae.load_state_dict(state_dict)
        vae.reconstruct_with_vae_map(sample_docs)

# 3. visualize