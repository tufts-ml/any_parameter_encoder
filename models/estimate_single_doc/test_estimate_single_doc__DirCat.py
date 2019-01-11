import argparse
import time
import numpy as np

import estimate_single_doc__DirCat

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--Ud', type=int, default=20,
        help="Number distinct vocab ids that appear in current document")
    parser.add_argument('--V', type=int, default=100,
        help="Total possible vocabulary size")
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument(
        '--lstep_max_iters',
        type=int,
        default=100)
    parser.add_argument(
        '--init_name',
        type=str,
        default='uniform_sample+cold_nefmap')
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--param_npz_path', type=str, default=None)
    args = parser.parse_args()

    lstep_kwargs = dict(**estimate_single_doc__DirCat.lstep_kwargs)
    lstep_kwargs['lstep_max_iters'] = args.lstep_max_iters
    if args.verbose:
        lstep_kwargs['verbose'] = True
        lstep_kwargs['very_verbose'] = True

    if args.param_npz_path is not None and os.path.exists(args.param_npz_path):
        raise NotImplemented("TODO polish this section if needed")
        Params = dict(np.load(args.param_npz_path).items())
        word_ct_d_U = Params.get('word_ct_d_U')
        topics_KUd = Params.get('topics_KV')[:, Params.get('word_id_d_U')]
        K, Ud = topics_KUd.shape
    else:
        K = args.K
        Ud = args.Ud
        V = args.V

        prng = np.random.RandomState(12342)

        # Create random topics that sum to one
        topics_KV = prng.rand(K, V)
        topics_KV /= np.sum(topics_KV, axis=1)[:,np.newaxis]

        # Draw word ids observed in the document unif at random
        word_id_d_U = prng.choice(np.arange(V), size=Ud, replace=False)

        # Draw random word counts from Unif({1, 2, 3})
        word_ct_d_U = prng.randint(low=1, high=3, size=Ud)
        word_ct_d_U = np.asarray(word_ct_d_U, dtype=np.float64)

    print("Applying K=%d topics to doc with Ud=%d uniq terms" % (K, Ud))
    print("")
    print("## Default kwargs")
    for key in sorted(lstep_kwargs):
        print("%-50s %s" % (key, lstep_kwargs[key]))
    print("")

    for method in [
            'VI_with_q=DirCat',
            ]:
        start_time = time.time()
        LP, info_dict = estimate_single_doc__DirCat.estimate_local_params_for_single_doc(
            word_id_d_U=word_id_d_U,
            word_ct_d_U=word_ct_d_U,
            topics_KV=topics_KV,
            alpha_K=args.alpha * np.ones(K),
            init_name=args.init_name,
            **lstep_kwargs)
        best_run_info = info_dict['info_dict__best_run']
        elapsed_time_sec = time.time() - start_time

        # theta_d_K: Parameter vector of Dirichlet approximate posterior
        theta_d_K = LP['theta_d_K']

        # E[pi_d_K] : expected value of pi ~ Dir(\theta)
        E_pi_d_K = theta_d_K / np.sum(theta_d_K)

        if E_pi_d_K.size > 8:
            top_ids = np.argsort(-1 * E_pi_d_K)[:8]
        else:
            top_ids = np.arange(K)
        print("RESULT %-20s : after %8.3f sec" % (method, elapsed_time_sec))
        print("    E[pi_d_K]    = %s" % ' '.join(['%.4f' % x for x in E_pi_d_K[top_ids]]))
        print("    n_iters      = %5d" % best_run_info['n_iters'])
        print("    did_converge = %d" % best_run_info['did_converge'])
        print("    alpha        = %.3f" % (args.alpha))