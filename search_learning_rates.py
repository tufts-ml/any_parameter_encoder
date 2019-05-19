import numpy as np
import torch

from pyro.optim import StepLR
from pyro.infer import SVI, Trace_ELBO

from datasets.load import load_toy_bars
from models.lda_lognorm import VAE_pyro
from common import train_save_VAE, save_loglik_to_csv, save_reconstruction_array


# global params
results_dir = 'mdreviews_lr'
results_file = 'results.csv'
n_topics = 20

# vae params
vae_params = {
    # 'n_hidden_layers': [1, 2, 5, 10, 20],
    # 'n_hidden_units': [10, 20, 50, 100, 200, 300, 500]
    # 'n_hidden_layers': [1, 2, 5],
    # 'n_hidden_units': [10, 20, 50, 100]
    'starting_learning_rate': [.1, .01, .001],
    'decay_steps': [1000, 10000, 100000]
}

num_examples = 10
sample_idx = list(range(10))

# dataset_names = ['train', 'valid', 'test', 'test_single', 'test_double', 'test_triple']
# datasets = load_toy_bars('toy_bars_10x10')
# data_tr = datasets[0]
# datasets[0] = data_tr[:1000]

dataset_names = ['train', 'valid', 'test']
train = np.load('datasets/mdreviews/train.npy')
valid = np.load('datasets/mdreviews/valid.npy')
test = np.load('datasets/mdreviews/test.npy')
datasets = [train, valid, test]
data_tr = datasets[0]
datasets[0] = data_tr[:792]
datasets[2] = datasets[2][:792]

# various VAEs, all with fixed decoders
init_params = {
    'topic_init': 'resources/mdreviews_topics.npy',
    'topic_trainable': False,
    'vocab_size': datasets[0].shape[1],
    'n_topics': n_topics,
    'results_dir': results_dir,
    'results_file': results_file,
    'inference': 'vae',
    'n_hidden_layers': 5,
    'n_hidden_units': 100
}

lda_orig = init_params.copy()
lda_orig.update({
    'model_name': 'lda_orig',
    'enc_topic_init': None,
    'enc_topic_trainable': True,
    'scale_trainable': False
})

lda_scale = init_params.copy()
lda_scale.update({
    'model_name': 'lda_scale',
    'enc_topic_init': None,
    'enc_topic_trainable': True,
    'scale_trainable': True
})

orig_model_configs = [
    lda_orig, lda_scale
]
#
# for model_config in orig_model_configs:
#     for starting_learning_rate in vae_params['starting_learning_rate']:
#         for decay_steps in vae_params['decay_steps']:
#             model_config.update({
#                 'starting_learning_rate': starting_learning_rate,
#                 'decay_steps': decay_steps
#             })
#             print(model_config)
#             train_save_VAE(data_tr, model_config, hallucinations=False)

model_configs = orig_model_configs[:]
for model_config in model_configs:
    model_config['model_name'] = model_config['model_name'] + '_hallucinations'

for model_config in model_configs:
    for starting_learning_rate in vae_params['starting_learning_rate']:
        for decay_steps in vae_params['decay_steps']:
            model_config.update({
                'starting_learning_rate': starting_learning_rate,
                'decay_steps': decay_steps
            })
            print(model_config)
            train_save_VAE(data_tr, model_config, hallucinations=True)