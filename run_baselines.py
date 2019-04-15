import os
import numpy as np
import torch

import pyro
from pyro.optim import StepLR
from pyro.infer import SVI, TraceMeanField_ELBO, Trace_ELBO
from pyro.infer.mcmc import MCMC, NUTS

from datasets.load import load_toy_bars
from models.lda_lognorm import VAE_pyro
from common import save_loglik_to_csv, save_reconstruction_array


# global params
datadir = 'toy_bars_10x10'
results_dir = 'problem_toy_bars'
results_file = 'results.csv'

num_examples = 10
sample_idx = list(range(10))

dataset_names = ['train', 'valid', 'test', 'test_single', 'test_double', 'test_triple']
datasets = load_toy_bars('toy_bars_10x10')
datasets[0] = datasets[0][:1000]

model_config = {
    'topic_init': 'resources/topics_10x10.npy',
    'topic_trainable': False,
    'vocab_size': 100,
    'n_topics': 18,
    'model_name': 'lda_scale',
    # we first define this for the model we load to instantiate the inference algorithms
    'results_dir': 'problem_toy_bars',
    'results_file': results_file,
    # these values don't matter because we're only taking the fixed decoder
    # # they're just needed to instantiate the model
    'n_hidden_units': 10,
    'n_hidden_layers': 2,
}


vae = VAE_pyro(**model_config)
state_dict = vae.load()
vae.load_state_dict(state_dict)
# for data_name, data in zip(dataset_names, datasets):
for data_name, data in zip(['valid'], [datasets[1]]):
    # pyro scheduler doesn't have any effect in the VAE case since we never take any optimization steps
    data = torch.from_numpy(data.astype(np.float32))
    pyro_scheduler = StepLR(
        {'optimizer': torch.optim.Adam, 'optim_args': {"lr": .1}, 'step_size': 10000, 'gamma': 0.95})
    vae_svi = SVI(vae.model, vae.encoder_guide, pyro_scheduler, loss=Trace_ELBO(), num_steps=100,
              num_samples=1000)
    svi = SVI(vae.model, vae.mean_field_guide, pyro_scheduler, loss=TraceMeanField_ELBO(), num_steps=100,
              num_samples=1000)
    mcmc = MCMC(NUTS(vae.model, adapt_step_size=True), num_samples=1000, warmup_steps=50)
    mcmc_lda = MCMC(NUTS(vae.lda_model, adapt_step_size=True), num_samples=1000, warmup_steps=50)
    for inference_name, inference in zip(['vae', 'svi', 'mcmc'], [vae_svi, svi, mcmc]):
        # for inference_name, inference in zip(['mcmc'], [mcmc]):
            # if (data_name == 'train') and (inference_name in ['svi', 'mcmc']):
            #     continue
            # run inference
        posterior = inference.run(data)
        latent_mean = np.mean([trace.nodes['latent']['value'][0].detach().numpy() for trace in posterior.exec_traces], axis=0)
        latent_std = np.std([trace.nodes['latent']['value'][0].detach().numpy() for trace in posterior.exec_traces], axis=0)
        print(inference_name)
        print(latent_mean)
        print(latent_std)
        with open(os.path.join(results_dir, 'posterior_mean_and_sd.txt'), 'a') as f:
            f.write(inference_name)
            f.write(repr(latent_mean))
            f.write(repr(latent_std))

        # model_config.update({
        #     'data_name': data_name,
        #     'inference': inference_name,
        #     # hack to get the various runs to save different files
        #     'n_hidden_layers': run,
        #     # we first define this for the model we load to instantiate the inference algorithms
        #     'results_dir': results_dir,
        # })
        # print(model_config)
        # save_reconstruction_array(vae, posterior, sample_idx, model_config)
        # for i in range(10):
        #     save_loglik_to_csv(data, vae.model, posterior, model_config, num_samples=10)
