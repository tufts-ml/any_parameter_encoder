import os
import numpy as np
import torch

from pyro.optim import StepLR
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from pyro.infer.mcmc import MCMC, NUTS

from datasets.load import load_toy_bars
from models.lda_lognorm import VAE_pyro
from common import train_save_VAE, save_loglik_to_csv, save_reconstruction_array
from visualization.reconstructions import plot_side_by_side_docs
from utils import softmax

# where to write the results
results_dir = 'experiments/vae_experiments/problem_toy_bars5'
results_file = 'results.csv'

# global params
n_topics = 18
num_examples = 10
sample_idx = list(range(10))


# toy bars data
# dataset_names = ['train', 'valid', 'test', 'test_single', 'test_double', 'test_triple']
datasets = load_toy_bars('toy_bars_10x10')
data_tr = datasets[0]
data_tr_single = data_tr[np.count_nonzero(data_tr, axis=1) <= 10]
data_tr_double = data_tr[np.count_nonzero(data_tr, axis=1) > 10]
datasets = [data_tr_single[:1000], data_tr_double[:1000]] + datasets[1:]
dataset_names = ['train_single', 'train_double', 'valid', 'test', 'test_single', 'test_double', 'test_triple']
# datasets[0] = data_tr[:1000]

# Amazon product reviews data
# dataset_names = ['train', 'valid', 'test']
# train = np.load('datasets/mdreviews/train.npy')
# valid = np.load('datasets/mdreviews/valid.npy')
# test = np.load('datasets/mdreviews/test.npy')
# datasets = [train, valid, test]
# data_tr = datasets[0]
# datasets[0] = data_tr[:792]
# datasets[2] = datasets[2][:792]


model_config = {
    # 'topic_init': 'resources/mdreviews_topics1.npy',
    'topic_init': 'resources/topics_10x10.npy',
    # 'topic_init': None,
    'topic_trainable': False,
    'vocab_size': datasets[0].shape[1],
    'n_topics': n_topics,
    'results_dir': results_dir,
    'results_file': results_file,
    'inference': 'vae',
    'model_name': 'lda_orig',
    'enc_topic_init': None,
    'enc_topic_trainable': True,
    'scale_trainable': False,
    'n_hidden_layers': 5,
    'n_hidden_units': 100,
    'n_samples': 1,
    'decay_rate': .5,
    'decay_steps': 1000,
    'starting_learning_rate': .01,
}


# train the VAE and save the weights
# train_save_VAE(data_tr, model_config, training_epochs=80, batch_size=30, hallucinations=False, tensorboard=True)
train_save_VAE(data_tr, model_config, training_epochs=80, batch_size=200, hallucinations=False, tensorboard=True)
# train_save_VAE(np.concatenate([data_tr_single[:10], data_tr_double[:10]]), model_config, training_epochs=80, hallucinations=False, tensorboard=True)
# load the VAE into pyro for evaluation
vae = VAE_pyro(**model_config)
state_dict = vae.load()
vae.load_state_dict(state_dict)
# topics = softmax(vae.decoder.topics.detach().numpy())
# np.save('topics.npy', topics)
# plot_side_by_side_docs(topics, os.path.join(results_dir, 'topics.pdf'))
for data_name, data in zip(dataset_names, datasets):
    data = torch.from_numpy(data.astype(np.float32))
    pyro_scheduler = StepLR({'optimizer': torch.optim.Adam, 'optim_args': {"lr": .1}, 'step_size': 10000, 'gamma': 0.95})
    # pyro scheduler doesn't have any effect in the VAE case since we never take any optimization steps
    vae_svi = SVI(vae.model, vae.encoder_guide, pyro_scheduler, loss=Trace_ELBO(), num_steps=100, num_samples=10)
    svi = SVI(vae.model, vae.mean_field_guide, pyro_scheduler, loss=TraceMeanField_ELBO(), num_steps=100, num_samples=10)
    # mcmc = MCMC(NUTS(vae.model, adapt_step_size=True), num_samples=1000, warmup_steps=50)
    for inference_name, inference in zip(['vae', 'svi'], [vae_svi, svi]):
        posterior = inference.run(data)
        model_config.update({
            'data_name': data_name,
            'inference': inference_name,
            'results_dir': results_dir,
        })
        print(model_config)
        save_reconstruction_array(vae, posterior, sample_idx, model_config)
        for i in range(10):
            # saves a separate row to the csv
            save_loglik_to_csv(data, vae.model, posterior, model_config, num_samples=10)
