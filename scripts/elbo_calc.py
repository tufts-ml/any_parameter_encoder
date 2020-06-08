import numpy as np
import scipy.stats

from scipy.special import logsumexp, softmax

np.set_printoptions(precision=3, suppress=1)

## Settings
## --------

K = 6 # num topics
V = 9 # num vocabs
n_docs = 50
n_words_per_doc = 500
seed = 42

alpha = 0.1
alpha_K = alpha * np.ones(K)

q_stddev = 0.1
n_mc_samples = 5

## Draw dataset from LDA model given bar topics
## --------------------------------------------

def draw_piDK_and_xDV(alpha_K, theta_KV, n_docs, n_words_per_doc, seed=0):
    K, V = theta_KV.shape
    prng = np.random.RandomState(seed)
    pi_DK = prng.dirichlet(alpha_K, size=(n_docs))
    x_DV = np.zeros((n_docs, V))
    for d in range(n_docs):
        x_DV[d] = prng.multinomial(n_words_per_doc, np.dot(pi_DK[d], theta_KV))
    return pi_DK, x_DV

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
    # '''
    np.set_printoptions(precision=3, suppress=1)
    print("Sanity check of translation of Dir prior to SoftmaxNormal")
    print("5 samples from the known Dir PRIOR")
    pi_SK = prng.dirichlet(alpha_K, size=5)
    print(pi_SK)
    print("5 samples from the estimated SoftmaxNormal PRIOR")
    h_SK = prng.randn(5, K) * prior_stddev_K[np.newaxis,:] + prior_loc_K[np.newaxis,:]
    print(softmax(h_SK, axis=1))
    print()
    # '''


    ## Estimate loc and scale parameters of q(pi)
    # when we invert softmax, location is ambiguous up to an additive constant,
    # so pick the one such that the sum of each row is zero
    loc_DK = np.log(pi_DK)
    loc_DK = loc_DK - np.mean(loc_DK, axis=1)[:,np.newaxis]
    scale_DK = sigma * np.ones((D, K))


    # Quick verify that our softmax-normal posteriors are reasonable
    # '''
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
    # '''

    ## Draw samples
    log_prior_pdf = np.zeros(D)
    log_lik_pdf = np.zeros(D)
    log_q_pdf = np.zeros(D)
    
    elbo_per_token_list = list()
    for s in range(n_mc_samples):
        h_samp_DK = prng.normal(loc_DK, scale_DK)
        pi_samp_DK = softmax(h_samp_DK, axis=1)

        for d in range(D):
            log_prior_pdf[d] = np.sum(scipy.stats.norm(prior_loc_K, prior_stddev_K).logpdf(h_samp_DK[d]))
            log_lik_pdf[d] = scipy.stats.multinomial(n=sum(x_DV[0]), p=np.dot(pi_samp_DK[d], theta_KV)).logpmf(x_DV[d])
            log_q_pdf[d] = np.sum(scipy.stats.norm(loc_DK[d], scale_DK[d]).logpdf(h_samp_DK[d]))

        elbo_per_token = np.sum(log_prior_pdf + log_lik_pdf - log_q_pdf) / np.sum(x_DV)
        elbo_per_token_list.append(elbo_per_token)

    return elbo_per_token_list

if __name__ == "__main__":

    ## Bar topics
    ## ----------

    proba_ontopic = 0.95 / 3
    proba_offtopic = 0.05 / (V - 3)

    theta_KV = proba_offtopic * np.ones((K, V))
    for kk, (v_start, v_stop) in enumerate([(0,3), (3,6), (6,9)]):
        theta_KV[kk, v_start:v_stop] = proba_ontopic
    for kk, words in enumerate([(0,3,6), (1,4,7), (2,5,8)]):
        theta_KV[3+kk, words] = proba_ontopic

    pi_DK, x_DV = draw_piDK_and_xDV(alpha_K, theta_KV, n_docs, n_words_per_doc, seed)
    print("%d MC samples of ELBO-per-token:" % n_mc_samples)
    print(np.asarray(calc_elbo_per_token(x_DV, alpha_K, theta_KV, pi_DK, q_stddev, n_mc_samples, seed=seed)))

    print("Unigram-logpmf-per-token:")
    phat_V = np.mean(x_DV, axis=0)
    phat_V = phat_V / np.sum(phat_V)
    unigram_logpmf_pertoken = 0.0
    for d in range(n_docs):
        unigram_logpmf_pertoken += scipy.stats.multinomial(n=sum(x_DV[d]), p=phat_V).logpmf(x_DV[d]) / np.sum(x_DV)
    print(np.asarray([unigram_logpmf_pertoken]))