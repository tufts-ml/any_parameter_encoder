import numpy as np
import torch

from pyro.optim import StepLR
from pyro.infer import SVI, Trace_ELBO

from datasets.load import load_toy_bars
from models.lda_lognorm import VAE_pyro
from common import train_save_VAE, save_loglik_to_csv, save_reconstruction_array


# global params
datadir = 'toy_bars_10x10'
results_dir = 'problem_toy_bars'
results_file = 'results.csv'
n_topics = 18

# vae params
vae_params = {
    'n_hidden_layers': [1, 2, 5, 10, 20],
    'n_hidden_units': [10, 20, 50, 100, 200, 300, 500]
}
# vae_params = {
#     'n_hidden_layers': [1, 2],
#     'n_hidden_units': [10, 20, 50, 100]
# }
num_examples = 10
sample_idx = list(range(10))

dataset_names = ['train', 'valid', 'test', 'test_single', 'test_double', 'test_triple']
datasets = load_toy_bars('toy_bars_10x10')
data_tr = datasets[0]

# various VAEs, all with fixed decoders
init_params = {
    'topic_init': 'resources/topics_10x10.npy',
    'topic_trainable': False,
    'vocab_size': 100,
    'n_topics': n_topics,
    'results_dir': results_dir,
    'results_file': results_file,
    'inference': 'vae',
}

lda_orig = init_params.copy()
lda_orig.update({
    'model_name': 'lda_orig_hallucinations',
    'enc_topic_init': None,
    'enc_topic_trainable': True,
    'scale_trainable': False
})

lda_scale = init_params.copy()
lda_scale.update({
    'model_name': 'lda_scale_hallucinations',
    'enc_topic_init': None,
    'enc_topic_trainable': True,
    'scale_trainable': True
})


lda_sym_fixed = init_params.copy()
lda_sym_fixed.update({
    'model_name': 'lda_sym_fixed_hallucinations',
    'enc_topic_init': 'resources/topics_10x10.npy',
    'enc_topic_trainable': False,
    'scale_trainable': False,
    'n_hidden_units': n_topics
})

lda_sym = init_params.copy()
lda_sym.update({
    'model_name': 'lda_sym_hallucinations',
    'enc_topic_init': 'resources/topics_10x10.npy',
    'enc_topic_trainable': True,
    'scale_trainable': False,
    'n_hidden_units': n_topics
})


lda_sym_fixed_with_scale = init_params.copy()
lda_sym_fixed_with_scale.update({
    'model_name': 'lda_sym_fixed_with_scale_hallucinations',
    'enc_topic_init': 'resources/topics_10x10.npy',
    'enc_topic_trainable': False,
    'scale_trainable': True,
    'n_hidden_units': n_topics
})

lda_sym_with_scale = init_params.copy()
lda_sym_with_scale.update({
    'model_name': 'lda_sym_with_scale_hallucinations',
    'enc_topic_init': 'resources/topics_10x10.npy',
    'enc_topic_trainable': True,
    'scale_trainable': True,
    'n_hidden_units': n_topics
})

model_configs = [
    lda_orig, lda_scale
]

model_configs_sym = [
    lda_sym_fixed, lda_sym, lda_sym_fixed_with_scale, lda_sym_with_scale
]

for model_config in model_configs:
    for n_hidden_layers in vae_params['n_hidden_layers']:
        for n_hidden_units in vae_params['n_hidden_units']:
            model_config.update({
                'n_hidden_layers': n_hidden_layers,
                'n_hidden_units': n_hidden_units,
            })
            print(model_config)
            train_save_VAE(data_tr, model_config, hallucinations=True)

for model_config in model_configs_sym:
    for n_hidden_layers in vae_params['n_hidden_layers']:
        model_config.update({
            'n_hidden_layers': n_hidden_layers,
        })
        print(model_config)
        train_save_VAE(data_tr, model_config, hallucinations=True)

for model_config in model_configs:
    for n_hidden_layers in vae_params['n_hidden_layers']:
        for n_hidden_units in vae_params['n_hidden_units']:
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
                    'n_hidden_layers': n_hidden_layers,
                    'n_hidden_units': n_hidden_units,
                    'data_name': data_name
                })
                print(model_config)
                save_reconstruction_array(vae, posterior, sample_idx, model_config)
                for i in range(10):
                    save_loglik_to_csv(data, vae.model, posterior, model_config, num_samples=10)


for model_config in model_configs_sym:
    for n_hidden_layers in vae_params['n_hidden_layers']:
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
                'n_hidden_layers': n_hidden_layers,
                'data_name': data_name
            })
            print(model_config)
            save_reconstruction_array(vae, posterior, sample_idx, model_config)
            for i in range(10):
                save_loglik_to_csv(data, vae.model, posterior, model_config, num_samples=10)
