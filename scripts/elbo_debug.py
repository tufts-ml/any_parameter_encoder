import numpy as np
from scripts.elbo_calc import calc_elbo_per_token


if __name__ == "__main__":
    pi_DK = np.load('data/toy_bars1/avi_template_loc.npy')
    x_DV = np.load('data/toy_bars1/docs_many_words.npy')[:5]
    theta_KV = np.load('data/toy_bars1/topics_many_words.npy')
    alpha_K = np.ones(20) * .1
    sigma = .1

    print(calc_elbo_per_token(x_DV, alpha_K, theta_KV, pi_DK, sigma, n_mc_samples=500, seed=0))