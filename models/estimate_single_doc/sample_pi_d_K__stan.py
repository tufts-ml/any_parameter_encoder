import numpy as np
import os
from pystan import StanModel
import cPickle
import time
import sys
import hashlib

_cache = dict()
stan_model_code_as_str = """

// DEFINE OBSERVED DATA
data {
    int<lower=1> K;                       // num topics
    int<lower=1> V;                       // num words
    int<lower=1> U;                       // num unique words in doc

    int<lower=1,upper=V> word_id_d_U[U];  // array of idxs into vocab
    int<lower=0> word_ct_d_U[U];          // array of counts

    vector<lower=0>[K] alpha_K;           // doc-topic prior concentration
    simplex[V] topics_KV[K];              // topic-word proba
}

// DEFINE FREE RANDOM VARIABLE
parameters {
    simplex[K] pi_d_K;                // doc-topic proba vector
}

// DEFINE LOG JOINT DENSITY AS PRIOR + LIKELIHOOD
model {
    pi_d_K ~ dirichlet(alpha_K);   // prior

    for (u in 1:U) {
        real log_prob_token_K[K];    // temp variable
        for (k in 1:K) {
            log_prob_token_K[k] = log(pi_d_K[k]) + log(topics_KV[k, word_id_d_U[u]]);
        }
        target += word_ct_d_U[u] * log_sum_exp(log_prob_token_K);
    }
}
"""

def sample_pi_d_K__stan(
        alpha_K=None,
        topics_KV=None,
        word_id_d_U=None,
        word_ct_d_U=None,
        n_samples=1000,
        n_burnin_samples=500,
        seed=42,
        verbose=False,
        ):
    global _cache
    if 'sm' in _cache:
        sm = _cache['sm']
    else:
        sm = load_compiled_stan_model(verbose=verbose)
        _cache['sm'] = sm

    K, V = topics_KV.shape
    word_id_d_U = np.asarray(word_id_d_U, dtype=np.int32)
    word_ct_d_U = np.asarray(word_ct_d_U, dtype=np.int32)
    # PACKAGE INTO DICT TO FEED STAN
    stan_feed_dict = dict(
        K=K,
        V=V,
        U=word_id_d_U.size,
        word_id_d_U=word_id_d_U + 1, # one based indexing
        word_ct_d_U=word_ct_d_U,
        alpha_K=alpha_K,
        topics_KV=topics_KV,
        )

    # Perform inference
    with suppress_stdout_stderr(verbose):
        fit = sm.sampling(
            data=stan_feed_dict,
            seed=int(seed),
            iter=n_samples + n_burnin_samples,
            warmup=n_burnin_samples,
            sample_file=None,
            diagnostic_file=None,
            verbose=verbose,
            chains=1)
    pi_d_SK = fit.extract(pars=('pi_d_K'))['pi_d_K']
    info_dict = dict(
        stan_fit_result=fit)
    return pi_d_SK, info_dict


def estimate_map_pi_d_K__stan(
        alpha_K=None,
        topics_KV=None,
        word_id_d_U=None,
        word_ct_d_U=None,
        init_pi_d_K=None,
        n_iters=100,
        seed=42,
        verbose=False,
        ):
    global _cache
    if 'sm' in _cache:
        sm = _cache['sm']
    else:
        sm = load_compiled_stan_model(verbose=verbose)
        _cache['sm'] = sm

    K, V = topics_KV.shape
    word_id_d_U = np.asarray(word_id_d_U, dtype=np.int32)
    word_ct_d_U = np.asarray(word_ct_d_U, dtype=np.int32)
    # PACKAGE INTO DICT TO FEED STAN
    stan_feed_dict = dict(
        K=K,
        V=V,
        U=word_id_d_U.size,
        word_id_d_U=word_id_d_U + 1, # one based indexing
        word_ct_d_U=word_ct_d_U,
        alpha_K=alpha_K,
        topics_KV=topics_KV,
        )

    # FYI init kwarg seems to be broken

    # Perform inference
    with suppress_stdout_stderr(verbose):
        fit = sm.optimizing(
            data=stan_feed_dict,
            seed=int(seed),
            iter=n_iters,
            algorithm='BFGS',
            sample_file=None,
            verbose=verbose,
            as_vector=False,
            )
    log_post_proba = fit['value']
    pi_d_K = fit['par']['pi_d_K']
    info_dict = dict(
        log_post_proba=log_post_proba,
        stan_fit_result=fit)
    return pi_d_K, info_dict


def load_compiled_stan_model(
        compiled_model_path=(
            "$SSCAPEROOT/bin/"
            + "compiled_stan_model__lda_single_doc.pkl"),
        compiled_md5_txt_path=(
            "$SSCAPEROOT/bin/"
            + "compiled_stan_model__lda_single_doc.src_code.md5.txt"),
        stan_model_code_as_str=stan_model_code_as_str,
        verbose=True,
        ):
    # Load compiled inference code
    compiled_model_path = os.path.expandvars(compiled_model_path)
    compiled_md5_txt_path = os.path.expandvars(compiled_md5_txt_path)
    md5_str = hashlib.md5(stan_model_code_as_str).hexdigest()
    if os.path.isfile(compiled_md5_txt_path):
        try:
            with open(compiled_md5_txt_path, 'r') as f:
                saved_md5_str = f.readline().strip()
        except IOError:
            saved_md5_str = None
    else:
        saved_md5_str = None
    sm = None
    if saved_md5_str == md5_str:
        # OK TO LOAD
        if verbose:
            print("Loading compiled Stan model from:\n%s" % (compiled_model_path))
        if os.path.isfile(compiled_model_path):
            fd = open(compiled_model_path, 'rb')
            sm = cPickle.load(fd)
            fd.close()
    if sm is None:
        # Compile
        sm = StanModel(
            model_name='lda_single_doc',
            model_code=stan_model_code_as_str,
            verbose=verbose)
        with open(compiled_model_path, 'wb') as f:
            cPickle.dump(sm, f)
        with open(compiled_md5_txt_path, 'w') as f:
            f.write(md5_str+"\n")
    return sm


class suppress_stdout_stderr(object):
    '''
    Context manager for full suppression of all stdout

    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
    
    This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    References
    ----------
    https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
    '''
    def __init__(self, verbose=True):
        self.verbose = bool(verbose)
        if not self.verbose:
            # Open a pair of null files
            self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
            # Save the actual stdout (1) and stderr (2) file descriptors.
            self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        if not self.verbose:
            # Assign the null pointers to stdout and stderr.
            os.dup2(self.null_fds[0], 1)
            os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        if not self.verbose:
            # Re-assign the real stdout/stderr back to (1) and (2)
            os.dup2(self.save_fds[0], 1)
            os.dup2(self.save_fds[1], 2)
            # Close all file descriptors
            for fd in self.null_fds + self.save_fds:
                os.close(fd)

if __name__ == '__main__':

    # DEFINE PARAMETERS
    K = 3
    V = 3
    topics_KV = 10.0 * np.eye(K) + 1.0
    topics_KV /= topics_KV.sum(axis=1)[:,np.newaxis]
    alpha_K = np.asarray([1.0, 3.0, 3.0]) + 1e-9


    print("alpha_K")
    print(alpha_K)
    if np.allclose(alpha_K, 1.0):
        print("MAP and ML should be the SAME")
    else:
        print("MAP and ML should be DIFFERENT")

    print("topics_KV")
    print(topics_KV)

    for ct in [0.0, 1.0, 10.0, 100.0]:
        word_id_d_U = np.zeros(1, dtype=np.int32)
        word_ct_d_U = ct * np.ones(1)
        
        x_d_V = np.zeros(V)
        x_d_V[word_id_d_U] = word_ct_d_U
        print("")
        print(x_d_V)

        pi_d_SK, info_dict = sample_pi_d_K__stan(
            word_id_d_U=word_id_d_U,
            word_ct_d_U=word_ct_d_U,
            alpha_K=alpha_K,
            topics_KV=topics_KV,
            n_samples=10000,
            n_burnin_samples=500,
            )
        print("mean pi_d_SK\n   %s" % (
            ' '.join(['%.4g' % a for a in pi_d_SK.mean(axis=0)])))

        map_pi_d_K, info_dict = estimate_map_pi_d_K__stan(
            word_id_d_U=word_id_d_U,
            word_ct_d_U=word_ct_d_U,
            alpha_K=alpha_K,
            topics_KV=topics_KV,
            n_iters=1000,
            )
        print(" MAP pi_d_SK\n   %s" % (
            ' '.join(['%.4g' % a for a in map_pi_d_K])))

        ml_pi_d_K, info_dict = estimate_map_pi_d_K__stan(
            word_id_d_U=word_id_d_U,
            word_ct_d_U=word_ct_d_U,
            alpha_K=np.ones(K),
            topics_KV=topics_KV,
            n_iters=1000,
            )
        print(" ML  pi_d_SK\n   %s" % (
            ' '.join(['%.4g' % a for a in ml_pi_d_K])))


