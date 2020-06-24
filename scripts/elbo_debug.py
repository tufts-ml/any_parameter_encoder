import numpy as np
import torch
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from model import APE
from pyro.optim import ExponentialLR

from scripts.elbo_calc import calc_elbo_per_token
from utils import softmax

if __name__ == "__main__":
    # pi_DK = softmax(np.load('data/toy_bars1/avi_template_loc.npy'))
    num_docs = 5
    x_DV = np.load('data/toy_bars1/docs_many_words.npy')[:num_docs]
    theta_KV = np.load('data/toy_bars1/topics_many_words.npy')
    alpha_K = np.ones(20) * .1
    # sigma = .1

    model_config = {
    'n_hidden_units': 100,
    'n_hidden_layers': 2,
    'alpha': .1,
    'vocab_size': 100,
    'n_topics': 20,
    'use_cuda': False,
    'architecture': 'template',
    'scale_type': 'sample',
    'skip_connections': False,
}
    vae = APE(**model_config)
    pyro_scheduler = ExponentialLR({'optimizer': torch.optim.Adam, 'optim_args': {"lr": .01}, 'gamma': 0.95})
    svi = SVI(vae.model, vae.mean_field_guide, pyro_scheduler, loss=TraceMeanField_ELBO(retain_graph=True), num_samples=100)
    avi = SVI(vae.model, vae.encoder_guide, pyro_scheduler, loss=TraceMeanField_ELBO(retain_graph=True), num_samples=100)
    docs = torch.from_numpy(x_DV).type(torch.FloatTensor)
    topics = torch.from_numpy(np.repeat(theta_KV, num_docs, axis=0)).type(torch.FloatTensor)
    print(topics.shape)
    print(avi.evaluate_loss(docs, topics) / x_DV.sum() / num_docs)
    pi_DK, var = vae.encoder.forward(docs, topics)
    print('var', var)
    pi_DK = pi_DK.detach().numpy()
    sigma = torch.sqrt(var).detach().numpy()[0]

    elbo_per_token_list = calc_elbo_per_token(x_DV, alpha_K, theta_KV, pi_DK, sigma, n_mc_samples=100, seed=0)
    print(elbo_per_token_list[:5])
    print(sum(elbo_per_token_list))
    print(sum(elbo_per_token_list)/x_DV.sum())