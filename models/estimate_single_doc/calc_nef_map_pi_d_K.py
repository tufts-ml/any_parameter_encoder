"""
calc_nef_map_pi_d_K.py : User-friend tool for MAP estimation of pi_d_K

API
---
Makes useful functions available:
* calc_nef_map_pi_d_K(...)

Validation
----------
$ python calc_nef_map_pi_d_K.py

Runs some diagnostic tests comparing different pi_d_K estimation methods.

"""
import argparse
import numpy as np
import time
import sys
import os

from calc_nef_map_pi_d_K__defaults import DefaultSingleDocOptimKwargs

## Load other modules
from calc_nef_map_pi_d_K__autograd import (
    calc_nef_map_pi_d_K__autograd)

try:
    from calc_nef_map_pi_d_K__numpy_linesearch import (
        calc_nef_map_pi_d_K__numpy_linesearch)
    HAS_LINESEARCH = True
except ImportError:
    HAS_LINESEARCH = False
    calc_nef_map_pi_d_K__numpy_linesearch = None

# Try to load tensorflow code
# Fall back on python
try:
    from calc_nef_map_pi_d_K__tensorflow import calc_nef_map_pi_d_K__tensorflow
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    calc_nef_map_pi_d_K__tensorflow = None

# Try to load cython code
# Fall back on python code
try:
    from calc_nef_map_pi_d_K__cython import calc_nef_map_pi_d_K__cython
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False
    calc_nef_map_pi_d_K__cython = None
try:
    from calc_nef_map_pi_d_K__cython_linesearch import (
        calc_nef_map_pi_d_K__cython_linesearch)
    HAS_CYTHON_LINESEARCH = True
except ImportError:
    HAS_CYTHON_LINESEARCH = False
    calc_nef_map_pi_d_K__cython_linesearch = None

def calc_nef_map_pi_d_K(
        word_id_d_U=None,
        word_ct_d_U=None,
        topics_KUd=None,
        topics_KV=None,
        alpha=None,
        nef_alpha=None,
        convex_alpha_minus_1=None,
        init_pi_d_K=None,
        method='autograd',
        pi_max_iters=DefaultSingleDocOptimKwargs['pi_max_iters'],
        pi_converge_thr=DefaultSingleDocOptimKwargs['pi_converge_thr'],
        pi_step_size=DefaultSingleDocOptimKwargs['pi_step_size'],
        pi_min_step_size=DefaultSingleDocOptimKwargs['pi_min_step_size'],
        pi_step_decay_rate=DefaultSingleDocOptimKwargs['pi_step_decay_rate'],
        pi_min_mass_preserved_to_trust_step=(
            DefaultSingleDocOptimKwargs['pi_min_mass_preserved_to_trust_step']),
        **kwargs):
    # Common preprocessing
    if topics_KUd is None:
        topics_KUd = topics_KV[:, word_id_d_U]

    # Precompute some useful things
    ct_topics_KUd = topics_KUd * word_ct_d_U[np.newaxis, :]
    K = topics_KUd.shape[0]

    # Parse alpha into natural EF alpha (so estimation is always convex)
    if convex_alpha_minus_1 is None:
        convex_alpha_minus_1 = make_convex_alpha_minus_1(
            alpha=alpha,
            nef_alpha=nef_alpha)
    assert convex_alpha_minus_1 < 1.0
    assert convex_alpha_minus_1 >= 0.0

    # Initialize as uniform vector over K simplex
    if init_pi_d_K is None:
        init_pi_d_K = np.ones(K) / float(K)
    else:
        init_pi_d_K = np.asarray(init_pi_d_K)
    assert init_pi_d_K.size == K

    if method.count("cython_linesearch"):
        if not HAS_CYTHON_LINESEARCH:
            raise ImportError("No compiled cython function: " + method)
        calc_pi_d_K = calc_nef_map_pi_d_K__cython_linesearch
    elif method.count("cython"):
        if not HAS_CYTHON:
            raise ImportError("No compiled cython function: " + method)
        calc_pi_d_K = calc_nef_map_pi_d_K__cython
    elif method.count("numpy_linesearch"):
        if not HAS_LINESEARCH:
            raise ImportError("No numpy function: " + method)
        calc_pi_d_K = calc_nef_map_pi_d_K__numpy_linesearch
    elif method.count("tensorflow"):
        if not HAS_TENSORFLOW:
            return None, None
        calc_pi_d_K = calc_nef_map_pi_d_K__tensorflow
    elif method.count("autograd"):
        calc_pi_d_K = calc_nef_map_pi_d_K__autograd
    else:
        raise ValueError("Unrecognized pi_d_K estimation method:" + method)
    pi_d_K, info = calc_pi_d_K(
        init_pi_d_K=init_pi_d_K,
        topics_KUd=topics_KUd,
        word_ct_d_U=np.asarray(word_ct_d_U, dtype=np.float64),
        ct_topics_KUd=ct_topics_KUd,
        convex_alpha_minus_1=convex_alpha_minus_1,
        pi_max_iters=int(pi_max_iters),
        pi_step_size=float(pi_step_size),
        pi_min_step_size=float(pi_min_step_size),
        pi_step_decay_rate=float(pi_step_decay_rate),
        pi_min_mass_preserved_to_trust_step=\
            float(pi_min_mass_preserved_to_trust_step),
        pi_converge_thr=float(pi_converge_thr),
        **kwargs)
    return pi_d_K, info


def make_convex_alpha_minus_1(alpha=None, nef_alpha=None):
    """ Convert provided alpha into its equivalent for convex MAP problem

    Returns
    -------
    convex_alpha_minus_1 : float
        will always be between 0 and 1
    """
    if nef_alpha is not None:
        nef_alpha = float(nef_alpha)
    elif alpha is not None:
        nef_alpha = float(alpha)
    else:
        raise ValueError("Need to define alpha or nef_alpha")
    assert isinstance(nef_alpha, float)

    # Now translate into convex_alpha_minus_1
    if nef_alpha > 1.0:
        convex_alpha_minus_1 = nef_alpha - 1.0
    else:
        convex_alpha_minus_1 = nef_alpha
    return convex_alpha_minus_1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--Ud', type=int, default=100)
    parser.add_argument('--nef_alpha', type=float, default=1.1)
    parser.add_argument(
        '--pi_max_iters',
        type=int,
        default=DefaultSingleDocOptimKwargs['pi_max_iters'])
    parser.add_argument(
        '--pi_step_size',
        type=float,
        default=DefaultSingleDocOptimKwargs['pi_step_size'])
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--param_npz_path', type=str, default=None)
    args = parser.parse_args()

    lstep_kwargs = dict(**DefaultSingleDocOptimKwargs)
    lstep_kwargs['pi_max_iters'] = args.pi_max_iters
    lstep_kwargs['pi_step_size'] = args.pi_step_size
    if args.verbose:
        lstep_kwargs['verbose'] = True
        lstep_kwargs['very_verbose'] = True

    if args.param_npz_path is not None and os.path.exists(args.param_npz_path):
        Params = dict(np.load(args.param_npz_path).items())
        nef_alpha = float(Params.get('nef_alpha', args.nef_alpha))
        word_ct_d_U = Params.get('word_ct_d_U')
        topics_KUd = Params.get('topics_KV')[:, Params.get('word_id_d_U')]

        K, Ud = topics_KUd.shape
    else:
        K = args.K
        Ud = args.Ud
        nef_alpha = args.nef_alpha

        prng = np.random.RandomState(12342)
        topics_KUd = prng.rand(K, Ud)
        topics_KUd /= np.sum(topics_KUd, axis=1)[:,np.newaxis]
        word_ct_d_U = prng.randint(low=1, high=3, size=Ud)
        word_ct_d_U = np.asarray(word_ct_d_U, dtype=np.float64)
    print("Applying K=%d topics to doc with Ud=%d uniq terms" % (K, Ud))
    print("nef_alpha = ", nef_alpha)
    print("")
    print("## Default kwargs")
    for key in sorted(lstep_kwargs):
        print "%-50s %s" % (key, lstep_kwargs[key])
    print("")

    for method in [
            'autograd',
            'tensorflow',
            #'cython',
            #'linesearch_numpy',
            #'linesearch_cython',
            ]:
        start_time = time.time()
        pi_d_K, info_dict = calc_nef_map_pi_d_K(
            word_ct_d_U=word_ct_d_U,
            topics_KUd=topics_KUd,
            nef_alpha=nef_alpha,
            method=method,
            **lstep_kwargs)
        if pi_d_K is None:
            print "SKIPPING %-20s" % (method)
            continue
        elapsed_time_sec = time.time() - start_time

        if pi_d_K.size > 8:
            top_ids = np.argsort(-1 * pi_d_K)[:8]
        else:
            top_ids = np.arange(K)
        print("RESULT %-20s : after %8.3f sec" % (method, elapsed_time_sec))
        print("    pi_d_K       = %s" % ' '.join(['%.4f' % x for x in pi_d_K[top_ids]]))
        print("    n_iters      = %5d" % info_dict['n_iters'])
        print("    did_converge = %d" % info_dict['did_converge'])
        print("    cur_L1_diff  = %.5f" % info_dict['cur_L1_diff'])
        print("    pi_step_size = %.5f" % info_dict['pi_step_size'])
        print("    n_restarts   = %d" % info_dict['n_restarts'])
        print("    nef_alpha    = %.3f" % (
            info_dict['convex_alpha_minus_1'] + 1.0))
