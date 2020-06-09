import numpy as np
import scipy.stats
import argparse
from scipy.special import logsumexp, softmax, gammaln

np.set_printoptions(precision=3, suppress=1)



parser = argparse.ArgumentParser()
## Generative model
parser.add_argument('--alpha', default=0.1, type=float)
## Inference
parser.add_argument('--q_stddev', default=0.1, type=float)
## Algorithm
parser.add_argument('--n_mc_samples', default=5, type=int)
parser.add_argument('--seed', default=42, type=int)

args = parser.parse_args()
locals().update(args.__dict__)

for k in args.__dict__.keys():
    print("--%s %s" % (k, args.__dict__[k]))

## Settings
## --------

x_DV = np.load('../../data/toy_bars_docs.npy')
theta_KV = np.load('../../data/toy_bars_docs_topics.npy')
pi_DK = np.load('../../data/toy_bars_docs_dist.npy')

K = pi_DK.shape[1] # num topics
V = x_DV.shape[1]  # num vocabs
n_docs = x_DV.shape[0]
n_words_per_doc = np.sum(x_DV, axis=1)

## Finalize generative model
alpha_K = alpha * np.ones(K)

## Finalize inference model


## Compute the ELBO
## ----------------

def calc_elbo_per_token(x_DV, alpha_K, theta_KV, pi_DK, sigma, n_mc_samples=500, seed=0):
    prng = np.random.RandomState(seed)
    D, V = x_DV.shape
    _, K = pi_DK.shape

    prior_loc_K = np.log(alpha_K) - 1.0/K * np.sum(np.log(alpha_K))
    prior_var_K = 1.0 / alpha_K * (1 - 2/K) + 1/(K*K) * np.sum(1.0/alpha_K)
    prior_stddev_K = np.sqrt(prior_var_K)

    # Quick verify that our prior setting is reasonable
    '''
    np.set_printoptions(precision=3, suppress=1)
    print("Sanity check of translation of Dir prior to SoftmaxNormal")
    print("5 samples from the known Dir PRIOR")
    pi_SK = prng.dirichlet(alpha_K, size=5)
    print(pi_SK)
    print("5 samples from the estimated SoftmaxNormal PRIOR")
    h_SK = prng.randn(5, K) * prior_stddev_K[np.newaxis,:] + prior_loc_K[np.newaxis,:]
    print(softmax(h_SK, axis=1))
    print()
    '''


    ## Estimate loc and scale parameters of q(pi)
    # when we invert softmax, location is ambiguous up to an additive constant,
    # so pick the one such that the sum of each row is zero
    loc_DK = np.log(pi_DK)
    loc_DK = loc_DK - np.mean(loc_DK, axis=1)[:,np.newaxis]
    scale_DK = sigma * np.ones((D, K))


    # Quick verify that our softmax-normal posteriors are reasonable
    '''
    for d in [0, 1, 2]:
        print("5 samples from the estimated Dir POSTERIOR for doc %d" % d)
        post_alpha_K = pi_DK[d] * n_words_per_doc + alpha_K
        print(post_alpha_K)
        pi_SK = prng.dirichlet(post_alpha_K, size=5)
        print(pi_SK)

        print("5 samples from the estimated SoftmaxNormal POSTERIOR for doc %d" % d)
        h_SK = prng.randn(5, K) * scale_DK[d][np.newaxis,:] + loc_DK[d][np.newaxis,:]
        print(softmax(h_SK, axis=1))
        print()
    '''
    logpmf_mult_const = (
        gammaln(1.0 + np.sum(x_DV, axis=1))
        - np.sum(gammaln(1.0 + x_DV), axis=1))
    #log_lik2_pdf = 1.0 * logpmf_mult_const

    ## Draw samples
    elbo_per_token_list = list()
    for s in range(n_mc_samples):
        log_prior_pdf = np.zeros(D)
        log_lik_pdf = np.zeros(D)
        log_q_pdf = np.zeros(D)

        h_samp_DK = prng.normal(loc_DK, scale_DK)
        pi_samp_DK = softmax(h_samp_DK, axis=1)
        log_lik2_pdf = logpmf_mult_const + np.sum(x_DV * np.log(1e-100 + np.dot(pi_samp_DK, theta_KV)), axis=1)

        for d in range(D):
            log_prior_pdf[d] = np.sum(scipy.stats.norm(prior_loc_K, prior_stddev_K).logpdf(h_samp_DK[d]))
            #log_lik_pdf[d] = scipy.stats.multinomial(n=n_words_per_doc[d], p=np.dot(pi_samp_DK[d], theta_KV)).logpmf(x_DV[d])
            log_q_pdf[d] = np.sum(scipy.stats.norm(loc_DK[d], scale_DK[d]).logpdf(h_samp_DK[d]))

        elbo_per_token = np.sum(log_prior_pdf + log_lik2_pdf - log_q_pdf) / np.sum(x_DV)
        elbo_per_token_list.append(elbo_per_token)
    return elbo_per_token_list

print("%d MC samples of ELBO-per-token:" % n_mc_samples)
print(np.asarray(calc_elbo_per_token(x_DV, alpha_K, theta_KV, pi_DK, q_stddev, n_mc_samples, seed=seed)))

print()
print("Unigram-logpmf-per-token:")
phat_V = np.mean(x_DV, axis=0)
phat_V = phat_V / np.sum(phat_V)
print(phat_V.reshape((5,5)))

unigram_logpmf_pertoken = 0.0
unigram_logpmf2_pertoken = 0.0
logpmf_mult_const = (
    gammaln(1.0 + np.sum(x_DV, axis=1))
    - np.sum(gammaln(1.0 + x_DV), axis=1))
for d in range(n_docs):
    unigram_logpmf2_pertoken += scipy.stats.multinomial(n=n_words_per_doc[d], p=phat_V).logpmf(x_DV[d]) / np.sum(x_DV)
    unigram_logpmf_pertoken += (logpmf_mult_const[d] + np.inner(x_DV[d], np.log(phat_V)) ) / np.sum(x_DV)
print(np.asarray([unigram_logpmf_pertoken]))
print(np.asarray([unigram_logpmf2_pertoken]))
