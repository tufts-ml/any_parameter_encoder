import autograd.numpy as np
import os
from autograd.scipy.special import gammaln, digamma
from autograd.scipy.misc import logsumexp
from collections import OrderedDict
from sklearn.externals import joblib

lstep_kwargs = dict(
    lstep_max_iters=100,
    lstep_converge_thr=0.0001,
    lstep_n_steps_between_print=5,
    lstep_coldstart_initname='cold_nefmap',
    )

loss_kwargs = dict(
    loss_prng=np.random.RandomState(11),
    loss_use_mc_objective=0,
    loss_n_mc_samples=1000,
    )

from calc_nef_map_pi_d_K import calc_nef_map_pi_d_K


def calc_q_logpdf_S__pi_d_SK(
        pi_d_SK=None,
        LP=None):
    return calc_dirichlet_logpdf_for_many_samples(pi_d_SK, LP['theta_d_K'])    

def init_empty_manydocsLP(n_docs=1, K=2):
    return dict(
        theta_DK=np.zeros((n_docs, K)),
        _equiv_pi_DK=np.zeros((n_docs, K)),
        )

def update_single_doc_in_manydocsLP(LP=None, d=0, theta_d_K=None, **kwargs):
    LP['theta_DK'][d] = theta_d_K
    # Just for monitoring...
    LP['_equiv_pi_DK'][d] = theta_d_K / np.sum(theta_d_K)
    return LP

def unpack_single_doc_from_manydocsLP(LP=None, d=0):
    LP_d = dict(
        theta_d_K=LP['theta_DK'][d],
        )
    try:
        LP_d['_equiv_pi_d_K'] = LP['_equiv_pi_DK'][d]
    except KeyError:
        pass
    return LP_d

def make_func_to_transform_loss_to_logpdfx():
    logpdfx_funcname = 'lower_bound_on_logpdfx'
    def transform(loss_val):
        return -1.0 * loss_val
    return transform, logpdfx_funcname

def make_pi_SK_generator_func__DirCat(
        LP=None,
        theta_d_K=None,
        seed=0, random_state=None, **kwargs):
    ''' Create function that generates pi samples from an LNKw posterior.

    Returns
    -------
    generate_pi_SK : function
        Calls to this function will produce pi_SK, a 2D array of samples
    '''
    if LP is None:
        LP = dict(theta_d_K=theta_d_K)
    if random_state is None:
        random_state = np.random.RandomState(int(seed))
    def generate_pi_SK(size=1, random_state=random_state):
        pi_SK = random_state.dirichlet(
            LP['theta_d_K'],
            size=size,
            )
        if not np.all(np.isfinite(pi_SK)):
            raise ValueError("NaN found")
        return pi_SK
    return generate_pi_SK

def calc_loss_for_single_doc(
        word_id_d_U=None,
        word_ct_d_U=None,
        topics_KV=None,
        alpha_K=None,
        LP=None,
        hLP=None,
        return_dict=False,
        loss_use_mc_objective=0,
        **kwargs):
    ''' Calculate loss the "long way", directly from theta_d_K, resp_d_UK

    Returns
    -------
    loss : float
        Equal to -1 * elbo(theta_d_K, resp_d_UK, doc_data)
    '''
    if int(loss_use_mc_objective) > 0:
        return calc_montecarlo_loss_for_single_doc__diffable_wrt_topics(
            word_id_d_U=word_id_d_U,
            word_ct_d_U=word_ct_d_U,
            topics_KV=topics_KV,
            alpha_K=alpha_K,
            LP=LP,
            **kwargs)

    # Fix input alignment / types
    word_id_d_U = np.ascontiguousarray(word_id_d_U, dtype=np.int32)
    word_ct_d_U = np.ascontiguousarray(word_ct_d_U, dtype=np.float64)
    alpha_K = np.ascontiguousarray(alpha_K, dtype=np.float64)
    topics_KU = topics_KV[:, word_id_d_U]

    assert 'theta_d_K' in LP
    if 'resp_d_UK' not in LP:
        # backtrack from theta_d_K to N_d_K
        N_d_K = np.maximum(LP['theta_d_K'] - alpha_K, 1e-100)
        LP, hLP = fill_local_params_for_single_doc__from_N_d_K(
            word_ct_d_U=word_ct_d_U,
            N_d_K=N_d_K,
            log_lik_d_UK=np.log(1e-100 + topics_KU).T,
            alpha_K=alpha_K)
    assert 'resp_d_UK' in LP

    if hLP is None:
        hLP = calc_helper_local_params_for_single_doc(
            word_ct_d_U=word_ct_d_U,
            resp_d_UK=LP['resp_d_UK'],
            theta_d_K=LP['theta_d_K'])

    log_norm_const_x_d = \
        gammaln(1.0 + np.sum(word_ct_d_U)) - \
        np.sum(gammaln(1.0 + word_ct_d_U))
    c_Dir_alpha = c_Dir_1D(alpha_K)

    loss = -1.0 * (
        log_norm_const_x_d
        + np.sum(hLP['C_d_KU'] * np.log(topics_KU))
        + np.sum(hLP['H_d_K'])
        + c_Dir_alpha - c_Dir_1D(LP['theta_d_K'])
        + np.inner(
            hLP['N_d_K'] + alpha_K - LP['theta_d_K'],
            hLP['E_log_pi_d_K'])
        )
    if return_dict:
        loss_dict = dict(
            E_logpdf_x=(
                log_norm_const_x_d
                + np.sum(hLP['C_d_KU'] * np.log(topics_KU))
                ),
            KL_q_to_prior=-1.0 * (
                np.sum(hLP['H_d_K'])
                + c_Dir_alpha - c_Dir_1D(LP['theta_d_K'])
                + np.inner(
                    hLP['N_d_K'] + alpha_K - LP['theta_d_K'],
                    hLP['E_log_pi_d_K'])
                ),
            H_q_z=np.sum(hLP['H_d_K']),
            c_Dir_alpha=c_Dir_alpha,
            log_norm_const_x_d=log_norm_const_x_d,
            )
        return loss, loss_dict
    else:
        return loss


def calc_montecarlo_loss_for_single_doc__diffable_wrt_topics(
        word_id_d_U=None,
        word_ct_d_U=None,
        topics_KV=None,
        alpha_K=None,
        LP=None,
        d=None,
        return_dict=False,
        loss_use_mc_objective=0,
        loss_n_mc_samples=1000,
        loss_prng=np.random.RandomState(42),
        **unused_kwargs):
    ''' Calculate loss via MonteCarlo estimate.

    Returns
    -------
    loss : float
        Equal to -1 * elbo(theta_d_K, resp_d_UK, doc_data)
    '''

    # Unpack settings
    S = int(loss_n_mc_samples)
    K = topics_KV.shape[0]
    assert K == alpha_K.size

    # Unpack local params
    if 'theta_d_K' in LP:
        theta_d_K = LP['theta_d_K']
    elif 'theta_DK' in LP:
        assert d is not None
        theta_d_K = LP['theta_DK'][d]

    # Sample
    pi_SK = loss_prng.dirichlet(theta_d_K, size=S)
    pi_SK = np.clip(pi_SK, 1e-13, 1.0 - K * 1e-13)

    topics_KU = topics_KV[:, word_id_d_U]

    # Evaluate expected log likelihood E_q[log p(x)]
    # log p(x | pi) = norm const of Mult logpmf  + main logpmf term
    log_p_x__norm_const = \
        gammaln(1.0 + np.sum(word_ct_d_U)) - \
        np.sum(gammaln(1.0 + word_ct_d_U))
    log_lik_d_SU = np.log(np.dot(pi_SK, topics_KU))
    logpdf_x_S = log_p_x__norm_const + np.dot(log_lik_d_SU, word_ct_d_U)

    logp_pi_S = calc_dirichlet_logpdf_for_many_samples(pi_SK, alpha_K)
    logq_pi_S = calc_dirichlet_logpdf_for_many_samples(pi_SK, theta_d_K)

    E_logpdf_x = np.mean(logpdf_x_S)
    E_logp_pi = np.mean(logp_pi_S)
    E_logq_pi = np.mean(logq_pi_S)

    loss = -1.0 * (
        E_logpdf_x + E_logp_pi - E_logq_pi)
    if return_dict:
        return loss, dict()
    else:
        return loss


def estimate_local_params_for_single_doc(
        word_id_d_U=None,
        word_ct_d_U=None,
        topics_KV=None,
        alpha_K=None,
        lstep_coldstart_initname=None,
        init_name=None,
        init_name_list=None,
        init_LP=None,
        init_P_d_K=None,
        prng=np.random,
        verbose=False,
        output_path=None,
        **lstep_kwargs):
    ''' Find LP for specific document that maximizes Dir-Cat ELBO.

    Args
    ----
    init_name_list : list of strings
        Possible inits to select from.

    Returns
    -------
    LP : dict
    hLP : dict
    info : dict
    '''
    # Fix input alignment / types
    word_id_d_U = np.ascontiguousarray(word_id_d_U, dtype=np.int32)
    word_ct_d_U = np.ascontiguousarray(word_ct_d_U, dtype=np.float64)
    alpha_K = np.ascontiguousarray(alpha_K, dtype=np.float64)
    lik_d_UK = np.ascontiguousarray(
        topics_KV[:, word_id_d_U].T, dtype=np.float64)
    log_lik_d_UK = np.log(1e-100 + lik_d_UK)

    K = alpha_K.size
    if init_name is not None:
        init_name_list = init_name.split("+")
    if init_name_list is None:
        init_name_list = [lstep_coldstart_initname]

    ## Now, initialize unnorm'd probability of each topic P_d_K
    if init_LP is not None:
        init_P_d_K_list = [init_LP['theta_d_K']]
    if not isinstance(init_P_d_K, list):
        init_P_d_K_list = [init_P_d_K for a in range(len(init_name_list))]
    else:
        init_P_d_K_list = init_P_d_K

    # Remember, init_P_d_K is a list
    assert len(init_P_d_K_list) == len(init_name_list)
    # Detect  warm start requested without provided warm array to start from,
    # and default to the prefered coldstart
    for aa in range(len(init_name_list)):
        name = init_name_list[aa]
        P_d_K = init_P_d_K_list[aa]
        if name.count('warm') and P_d_K is None:
            init_name_list[aa] = lstep_coldstart_initname

    log_norm_const_x_d = \
        gammaln(1.0 + np.sum(word_ct_d_U)) - \
        np.sum(gammaln(1.0 + word_ct_d_U))
    best_loss_val = np.inf
    best_N_d_K = None
    best_info = None

    # Create dict of info we'll return about all optimization runs
    info_dict = dict()
    info_dict['info_dict__all_runs'] = OrderedDict()
    info_dict['trace_dict_by_init'] = OrderedDict()
    trace_dict_by_init = OrderedDict()
    for init_name in init_name_list:
        if init_name.count("_x") > 0:
            init_name_fields = init_name.split("_x")
            init_name = init_name_fields[0]
            n_reps = int(init_name_fields[1])
        else:
            n_reps = 1
        for rep in xrange(n_reps):
            if n_reps > 1:
                rep_key = init_name + "_rep%03d" % (rep+1)
            else:
                rep_key = init_name
            init_P_d_K = make_init_P_d_K(
                init_name, prng, K, word_ct_d_U, lik_d_UK, alpha_K, init_P_d_K_list)
            if verbose:
                pprint__N_d_K(init_P_d_K, "init")

            cur_N_d_K, cur_info = \
                optimize_N_d_K_for_single_doc__vb_coord_descent(
                    word_ct_d_U=word_ct_d_U,
                    log_lik_d_UK=log_lik_d_UK,
                    lik_d_UK=lik_d_UK,
                    alpha_K=alpha_K,
                    init_P_d_K=init_P_d_K,
                    verbose=verbose,
                    **lstep_kwargs)
            cur_info['init_name'] = rep_key
            cur_loss_val = calc_simplified_loss_for_single_doc__from_N_d_K(
                word_ct_d_U=word_ct_d_U,
                log_lik_d_UK=log_lik_d_UK,
                alpha_K=alpha_K,
                N_d_K=cur_N_d_K,
                log_norm_const_x_d=log_norm_const_x_d)
            if verbose:
                pprint__N_d_K(cur_N_d_K, "final", cur_loss_val)

            # Convert loss values into estimates or bounds of logpdfx
            transform, funcname = make_func_to_transform_loss_to_logpdfx()
            cur_info['logpdfx_value'] = transform(cur_loss_val)
            cur_info['logpdfx_funcname'] = funcname

            # Add current run to overall info_dict
            if n_reps > 1:
                rep_key = init_name + "_rep%03d" % (rep+1)
            else:
                rep_key = init_name
            if 'trace_loss' in cur_info:
                cur_info['trace_loss'] = \
                    cur_info['trace_loss'] - log_norm_const_x_d
                cur_info['trace_logpdfx'] = transform(cur_info['trace_loss'])
                trace_dict_by_init[rep_key] = cur_info

            # Save the current run's info to a joblib dump file
            # Can be read again with joblib.load(...)
            if isinstance(output_path, str) and os.path.exists(output_path):
                cur_output_file = os.path.join(
                    output_path, '%03d_run_info.dump' % rep)
                joblib.dump(
                    cur_info,
                    cur_output_file,
                    compress=1)
                cur_info['output_path'] = output_path
                cur_info['output_dump_fpath'] = cur_output_file

            # Update 'best' run if current run beats it
            if cur_loss_val < best_loss_val - 1e-6:
                best_loss_val = cur_loss_val
                best_N_d_K = cur_N_d_K
                best_info = cur_info
                if verbose:
                    print "best: %s" % init_name
            elif cur_loss_val < best_loss_val + 1e-6:
                if verbose:
                    print "tied: %s" % init_name


    if verbose:
        print ""
    info_dict['loss_val__best_run'] = best_loss_val
    info_dict['info_dict__best_run'] = best_info
    info_dict['trace_dict_by_init'] = trace_dict_by_init

    # Fill out complete local parameters
    # given the optimal N_d_K value
    LP, hLP = fill_local_params_for_single_doc__from_N_d_K(
        word_ct_d_U=word_ct_d_U,
        N_d_K=best_N_d_K,
        log_lik_d_UK=log_lik_d_UK,
        alpha_K=alpha_K)
    info_dict['info_dict__best_run']['LP'] = LP
    info_dict['info_dict__best_run']['helper_LP'] = hLP
    info_dict['info_dict__best_run']['generate_pi_SK'] = \
        make_pi_SK_generator_func__DirCat(LP)
    return LP, info_dict

def fill_local_params_for_single_doc__from_N_d_K(
        word_ct_d_U=None,
        N_d_K=None,
        log_lik_d_UK=None,
        alpha_K=None,
        **unused_kwargs):
    theta_d_K = N_d_K + alpha_K
    E_log_pi_d_K = digamma(theta_d_K) - digamma(np.sum(theta_d_K))
        
    resp_d_UK = log_lik_d_UK + E_log_pi_d_K[np.newaxis,:]
    resp_d_UK -= np.max(resp_d_UK, axis=1)[:,np.newaxis]
    np.exp(resp_d_UK, out=resp_d_UK)
    resp_d_UK /= resp_d_UK.sum(axis=1)[:,np.newaxis]
    helperLP = calc_helper_local_params_for_single_doc(
        word_ct_d_U=word_ct_d_U,
        resp_d_UK=resp_d_UK,
        E_log_pi_d_K=E_log_pi_d_K)
    LP = dict(
        theta_d_K=theta_d_K,
        resp_d_UK=resp_d_UK)
    return LP, helperLP

def calc_helper_local_params_for_single_doc(
        word_ct_d_U=None,
        resp_d_UK=None,
        E_log_pi_d_K=None,
        theta_d_K=None):
    N_d_K = np.dot(word_ct_d_U, resp_d_UK)
    C_d_KU = resp_d_UK.T * word_ct_d_U[np.newaxis,:]

    # v2 : faster / mem efficient
    logresp_UK = np.log(1e-100 + resp_d_UK)
    np.multiply(logresp_UK, resp_d_UK, out=logresp_UK)
    H_d_K = np.dot(word_ct_d_U, logresp_UK)
    H_d_K *= -1.0
    # v1 : slow
    # H2_d_K = -1.0 * np.dot(
    #    word_ct_d_U, resp_d_UK * np.log(1e-100 + resp_d_UK))
    #assert np.allclose(H_d_K, H2_d_K)
    hLP = dict(
        N_d_K=N_d_K,
        C_d_KU=C_d_KU,
        H_d_K=H_d_K)
    if E_log_pi_d_K is not None:
        hLP['E_log_pi_d_K'] = E_log_pi_d_K
    else:
        hLP['E_log_pi_d_K'] = digamma(theta_d_K) - digamma(np.sum(theta_d_K))
    return hLP


def make_init_P_d_K(
        init_name, prng, K, word_ct_d_U, lik_d_UK, alpha_K, init_P_d_K_list):
    p_d_K = None
    if init_name.count('warm'):
        p_d_K = init_P_d_K_list.pop()
    elif init_name.count('fromscratchnaive'):
        p_d_K = prng.dirichlet(np.ones(K))
    elif init_name.count('uniform_sample'):
        p_d_K = prng.dirichlet(np.ones(K))
    elif init_name.count('prior_sample'):
        p_d_K = prng.dirichlet(alpha_K)
    elif init_name.count('cold_nefmap'):
        p_d_K, info = calc_nef_map_pi_d_K(
            word_ct_d_U=word_ct_d_U,
            topics_KUd=lik_d_UK.T,
            nef_alpha=1.0 + alpha_K[0],
            converge_thr=0.00001)
    elif init_name.count("cold_alpha"):
        p_d_K = alpha_K / np.sum(alpha_K) #np.zeros(K, dtype=alpha_K.dtype)
    else:
        raise ValueError("Unrecognized vb lstep_init_name: " + init_name)
    return np.ascontiguousarray(p_d_K, dtype=np.float64)

def pprint__N_d_K(N_d_K, label='', loss=None):
    if loss:
        print \
            "%12s" % label, \
            ' '.join(['%7.2f' % a for a in N_d_K]), \
            "%.7e" % loss
    else:
        print "%12s" % label, ' '.join(['%7.2f' % a for a in N_d_K])

def optimize_N_d_K_for_single_doc__vb_coord_descent(
        word_ct_d_U=None,
        log_lik_d_UK=None,
        lik_d_UK=None,
        alpha_K=None,
        init_theta_d_K=None,
        init_N_d_K=None,
        init_P_d_K=None,
        lstep_converge_thr=0.0001,
        lstep_max_iters=100,
        lstep_n_steps_between_print=5,
        lstep_n_steps_between_check=5,
        verbose=False,
        **unused_kwargs):
    # Parse keyword args
    lstep_max_iters = int(lstep_max_iters)
    lstep_converge_thr = float(lstep_converge_thr)
    lstep_n_steps_between_check = int(lstep_n_steps_between_check)
    lstep_n_steps_between_print = int(lstep_n_steps_between_print)

    if log_lik_d_UK is None:
        log_lik_d_UK = np.log(1e-100 + lik_d_UK)
    P_d_K = np.zeros_like(alpha_K)
    sumresp_U = np.zeros_like(word_ct_d_U)
    if init_P_d_K is not None:
        P_d_K[:] = init_P_d_K
        N_d_K = np.zeros_like(alpha_K)
        np.dot(lik_d_UK, P_d_K, out=sumresp_U)
        np.dot(word_ct_d_U / sumresp_U, lik_d_UK, out=N_d_K)
        N_d_K *= P_d_K
    elif init_theta_d_K is not None:
        N_d_K = np.maximum(init_theta_d_K - alpha_K, 1e-10)
    elif init_N_d_K is not None:
        N_d_K = init_N_d_K

    prev_N_d_K = np.zeros_like(N_d_K)
    digamma_sumtheta_d = digamma(np.sum(alpha_K) + np.sum(word_ct_d_U))

    do_trace = lstep_n_steps_between_print > 0
    if do_trace:
        iter_list = list()
        loss_list = list()

    converge_dist = np.inf
    local_iter = 0
    for local_iter in range(1, 1+lstep_max_iters):
        if do_trace:
            loss = calc_simplified_loss_for_single_doc__from_N_d_K(
                word_ct_d_U=word_ct_d_U,
                log_lik_d_UK=log_lik_d_UK,
                alpha_K=alpha_K,
                N_d_K=N_d_K)
            iter_list.append(local_iter)
            loss_list.append(loss)                 

        np.add(N_d_K, alpha_K, out=P_d_K)
        digamma(P_d_K, out=P_d_K)
        np.subtract(P_d_K, digamma_sumtheta_d, out=P_d_K)
        np.exp(P_d_K, out=P_d_K)
        np.dot(lik_d_UK, P_d_K, out=sumresp_U)
        # Update DocTopicCounts
        np.dot(word_ct_d_U / sumresp_U, lik_d_UK, out=N_d_K)
        N_d_K *= P_d_K

        is_every = local_iter % int(lstep_n_steps_between_print) == 0
        is_last = local_iter == lstep_max_iters
        if verbose and (is_every or is_last):
            pprint__N_d_K(N_d_K, 'it %4d/%4d' % (local_iter, lstep_max_iters))

        if local_iter % int(lstep_n_steps_between_check) == 0:
            converge_dist = np.sum(np.abs(N_d_K - prev_N_d_K))
            if converge_dist < lstep_converge_thr:
                break
        prev_N_d_K[:] = N_d_K

    opt_info = dict(
        n_iters=local_iter,
        max_iters=lstep_max_iters,
        did_converge = converge_dist < lstep_converge_thr,
        converge_thr = lstep_converge_thr,
        converge_dist = converge_dist,
        )

    if do_trace:
        opt_info['trace_steps'] = np.asarray(iter_list)
        opt_info['trace_loss'] = np.asarray(loss_list)

    return N_d_K, opt_info


def calc_simplified_loss_for_single_doc__from_N_d_K(
        word_ct_d_U=None,
        log_lik_d_UK=None,
        alpha_K=None,
        N_d_K=None,
        log_norm_const_x_d=0.0):
    ''' Compute loss up to const that doesnt depend on local parameters.

    Returns
    -------
    loss_val : scalar float
        Equal to -1 * elbo + constant
    '''
    theta_d_K = N_d_K + alpha_K
    E_log_pi_d_K = digamma(theta_d_K) - digamma(np.sum(theta_d_K))
    log_resp_d_UK = log_lik_d_UK + E_log_pi_d_K[np.newaxis,:]
    return -1.0 * (
        log_norm_const_x_d
        + np.inner(word_ct_d_U, logsumexp(log_resp_d_UK, axis=1))
        + c_Dir_1D(alpha_K) - c_Dir_1D(theta_d_K)
        + np.inner(alpha_K - theta_d_K, E_log_pi_d_K)
        )

def c_Dir_1D(alpha_K):
    return gammaln(np.sum(alpha_K)) - np.sum(gammaln(alpha_K))

def calc_dirichlet_logpdf_for_many_samples(pi_SK, alpha_K):
    log_norm_const = gammaln(np.sum(alpha_K)) - np.sum(gammaln(alpha_K))
    return log_norm_const + np.inner(np.log(pi_SK), alpha_K - 1.0)

