import numpy as np
import torch

from pyro.optim import StepLR
from pyro.infer import SVI, Trace_ELBO

from datasets.load import load_toy_bars
from models.lda_lognorm import VAE_pyro
from common import train_save_VAE, save_loglik_to_csv, save_reconstruction_array


# global params
results_dir = 'problem_toy_bars'
results_file = 'results_corrected_full.csv'
n_topics = 18

# vae params
vae_params = {
    # 'n_hidden_layers': # [1, 2, 5, 10, 20],
    'n_hidden_units': [10, 20, 50, 100, 200, 300, 500],
    # 'n_hidden_layers': [1, 2],
    'n_hidden_layers': [5],
    # 'n_hidden_units': [10, 20, 50, 100]
}

num_examples = 10
sample_idx = list(range(10))

# dataset_names = ['train', 'valid', 'test', 'test_single', 'test_double', 'test_triple']
datasets = load_toy_bars('toy_bars_10x10')
data_tr = datasets[0]
# datasets[0] = data_tr[:1000]
data_tr_single = data_tr[np.count_nonzero(data_tr, axis=1) <= 10]
data_tr_double = data_tr[np.count_nonzero(data_tr, axis=1) > 10]
datasets = [data_tr_single[:1000], data_tr_double[:1000]]
dataset_names = ['train_single', 'train_double']

# dataset_names = ['train', 'valid', 'test']
# train = np.load('datasets/mdreviews/train.npy')
# valid = np.load('datasets/mdreviews/valid.npy')
# test = np.load('datasets/mdreviews/test.npy')
# datasets = [train, valid, test]
# data_tr = datasets[0]
# datasets[0] = data_tr[:792]
# datasets[2] = datasets[2][:792]

# various VAEs, all with fixed decoders
init_params = {
    'topic_init': 'resources/topics_10x10.npy', # 'resources/mdreviews_topics.npy',
    'topic_trainable': False,
    'vocab_size': datasets[0].shape[1],
    'n_topics': n_topics,
    'results_dir': results_dir,
    'results_file': results_file,
    'inference': 'vae',
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

lda_orig_hallucinations = init_params.copy()
lda_orig_hallucinations.update({
    'model_name': 'lda_orig_hallucinations',
    'enc_topic_init': None,
    'enc_topic_trainable': True,
    'scale_trainable': False
})

lda_scale_hallucinations = init_params.copy()
lda_scale_hallucinations.update({
    'model_name': 'lda_scale_hallucinations',
    'enc_topic_init': None,
    'enc_topic_trainable': True,
    'scale_trainable': True
})

# orig_model_configs = [
#     lda_orig, lda_scale, lda_orig_hallucinations, lda_scale_hallucinations
# ]
orig_model_configs = [lda_orig]

# lda_sym_fixed = init_params.copy()
# lda_sym_fixed.update({
#     'model_name': 'lda_sym_fixed_hallucinations',
#     'enc_topic_init': 'resources/topics_10x10.npy',
#     'enc_topic_trainable': False,
#     'scale_trainable': False,
#     'n_hidden_units': n_topics
# })
#
# lda_sym = init_params.copy()
# lda_sym.update({
#     'model_name': 'lda_sym_hallucinations',
#     'enc_topic_init': 'resources/topics_10x10.npy',
#     'enc_topic_trainable': True,
#     'scale_trainable': False,
#     'n_hidden_units': n_topics
# })
#
#
# lda_sym_fixed_with_scale = init_params.copy()
# lda_sym_fixed_with_scale.update({
#     'model_name': 'lda_sym_fixed_with_scale_hallucinations',
#     'enc_topic_init': 'resources/topics_10x10.npy',
#     'enc_topic_trainable': False,
#     'scale_trainable': True,
#     'n_hidden_units': n_topics
# })
#
# lda_sym_with_scale = init_params.copy()
# lda_sym_with_scale.update({
#     'model_name': 'lda_sym_with_scale_hallucinations',
#     'enc_topic_init': 'resources/topics_10x10.npy',
#     'enc_topic_trainable': True,
#     'scale_trainable': True,
#     'n_hidden_units': n_topics
# })


# model_configs_sym = [
#     lda_sym_fixed, lda_sym, lda_sym_fixed_with_scale, lda_sym_with_scale
# ]

# for model_config in orig_model_configs:
#     for n_hidden_layers in vae_params['n_hidden_layers']:
#         for n_hidden_units in vae_params['n_hidden_units']:
#             model_config.update({
#                 'n_hidden_layers': n_hidden_layers,
#                 'n_hidden_units': n_hidden_units,
#             })
#             print(model_config)
#             train_save_VAE(data_tr, model_config, hallucinations=False)
#
# model_configs = [config.copy() for config in orig_model_configs]
# for model_config in model_configs:
#     model_config['model_name'] = model_config['model_name'] + '_hallucinations'
#
# for model_config in model_configs:
#     for n_hidden_layers in vae_params['n_hidden_layers']:
#         for n_hidden_units in vae_params['n_hidden_units']:
#             if n_hidden_layers == 10 and n_hidden_units == 10:
#                 continue
#             model_config.update({
#                 'n_hidden_layers': n_hidden_layers,
#                 'n_hidden_units': n_hidden_units,
#             })
#             print(model_config)
#             train_save_VAE(data_tr, model_config, hallucinations=True)

for model_config in orig_model_configs:
    for n_hidden_layers in vae_params['n_hidden_layers']:
        for n_hidden_units in vae_params['n_hidden_units']:
            model_config.update({
                'n_hidden_layers': n_hidden_layers,
                'n_hidden_units': n_hidden_units,
            })
            vae = VAE_pyro(**model_config)
            state_dict = vae.load()
            vae.load_state_dict(state_dict)
            for data_name, data in zip(dataset_names, datasets):
                # pyro scheduler doesn't have any effect in the VAE case since we never take any optimization steps
                pyro_scheduler = StepLR(
                    {'optimizer': torch.optim.Adam, 'optim_args': {"lr": .1}, 'step_size': 10000, 'gamma': 0.95})
                vae_svi = SVI(vae.model, vae.encoder_guide, pyro_scheduler, loss=Trace_ELBO(), num_steps=0)
                data = torch.from_numpy(data.astype(np.float32))
                posterior = vae_svi.run(data)
                model_config.update({
                    'data_name': data_name
                })
                print(model_config)
                save_reconstruction_array(vae, posterior, sample_idx, model_config)
                for i in range(10):
                    save_loglik_to_csv(data, vae.model, posterior, model_config, num_samples=10)

# for model_config in [lda_scale]:
#     for n_hidden_layers in vae_params['n_hidden_layers']:
#         for n_hidden_units in vae_params['n_hidden_units']:
#             if n_hidden_units == 10 and n_hidden_layers == 10:
#                 continue
#             model_config.update({
#                 'n_hidden_layers': n_hidden_layers,
#                 'n_hidden_units': n_hidden_units,
#                 'model_name': 'lda_scale_hallucinations'
#             })
#             vae = VAE_pyro(**model_config)
#             state_dict = vae.load()
#             vae.load_state_dict(state_dict)
#             for data_name, data in zip(dataset_names, datasets):
#                 # pyro scheduler doesn't have any effect in the VAE case since we never take any optimization steps
#                 pyro_scheduler = StepLR(
#                     {'optimizer': torch.optim.Adam, 'optim_args': {"lr": .1}, 'step_size': 10000, 'gamma': 0.95})
#                 vae_svi = SVI(vae.model, vae.encoder_guide, pyro_scheduler, loss=Trace_ELBO(), num_steps=0)
#                 data = torch.from_numpy(data.astype(np.float32))
#                 posterior = vae_svi.run(data)
#                 model_config.update({
#                     'data_name': data_name
#                 })
#                 print(model_config)
#                 save_reconstruction_array(vae, posterior, sample_idx, model_config)
#                 for i in range(10):
#                     save_loglik_to_csv(data, vae.model, posterior, model_config, num_samples=10)