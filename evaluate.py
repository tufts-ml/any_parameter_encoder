import os
import csv
import math
import numpy as np
import itertools
import time
import logging

import torch

import pyro
from pyro.distributions.util import logsumexp
from pyro.infer.abstract_infer import TracePredictive
from pyro.infer.util import torch_item
from pyro.optim import StepLR
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from pyro.infer.mcmc import MCMC
from pyro.util import torch_isnan
import pyro.poutine as poutine


class TimedSVI(SVI):
    def __init__(self, *args, **kwargs):
        self.run_times = []
        super(TimedSVI, self).__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        # we do not include tracelist generation in the time
        # since the posterior is already fully specified before then
        start = time.time()
        if self.num_steps > 0:
            with poutine.block():
                for i in range(self.num_steps):
                    self.step(*args, **kwargs)
        end = time.time()
        self.run_times.append(end - start)
        return super(SVI, self).run(*args, **kwargs)
    

class TimedAVI(SVI):
    def __init__(self, *args, encoder=None, **kwargs):
        self.run_times = []
        self.encoder = encoder
        super(TimedAVI, self).__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        # we do not include tracelist generation in the time
        # since the posterior is already fully specified before then
        start = time.time()
        self.encoder.forward(*args)
        end = time.time()
        self.run_times.append(end - start)
        return super(SVI, self).run(*args, **kwargs)


class TimedMCMC(MCMC):
    def __init__(self, *args, **kwargs):
        self.run_times = []
        super(TimedMCMC, self).__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        start = time.time()
        posterior = super(TimedMCMC, self).run(*args, **kwargs)
        end = time.time()
        self.run_times.append(end - start)
        return posterior


def evaluate_log_predictive_density(posterior_predictive_traces):
    """
    Evaluate the log probability density of observing the unseen data
    given a model and empirical distribution over the parameters.
    """
    trace_log_pdf = []
    for tr in posterior_predictive_traces.exec_traces:
        trace_log_pdf.append(tr.log_prob_sum())
    # Use LogSumExp trick to evaluate $log(1/num_samples \sum_i p(new_data | \theta^{i})) $,
    # where $\theta^{i}$ are parameter samples from the model's posterior.
    posterior_pred_density = logsumexp(torch.stack(trace_log_pdf), dim=-1) - math.log(len(trace_log_pdf))
    return posterior_pred_density


def get_posterior_predictive_density(data, topics, model, posterior, num_samples=10):
    posterior_predictive = TracePredictive(model, posterior, num_samples=num_samples)
    posterior_predictive_traces = posterior_predictive.run(data, topics)
    # get the posterior predictive log likelihood
    posterior_predictive_density = evaluate_log_predictive_density(posterior_predictive_traces)
    posterior_predictive_density = float(posterior_predictive_density.detach().numpy())
    return posterior_predictive_density



import weakref

import pyro
import pyro.ops.jit
from pyro.distributions.util import is_identically_zero
from pyro.infer.elbo import ELBO
from pyro.infer.enum import get_importance_trace
from pyro.infer.util import MultiFrameTensor, get_plate_stacks, is_validation_enabled, torch_item
from pyro.util import check_if_enumerated, warn_if_nan


def _compute_log_r(model_trace, guide_trace):
    log_r = MultiFrameTensor()
    stacks = get_plate_stacks(model_trace)
    for name, model_site in model_trace.nodes.items():
        if model_site["type"] == "sample":
            log_r_term = model_site["log_prob"]
            if not model_site["is_observed"]:
                log_r_term = log_r_term - guide_trace.nodes[name]["log_prob"]
            log_r.add((stacks[name], log_r_term.detach()))
    return log_r


class Trace_ELBO_no_KL(Trace_ELBO):

    def _differentiable_loss_particle(self, model_trace, guide_trace):
        """
        Difference from Trace_ELBO is that we do not have the entropy term
        """
        elbo_particle = 0
        surrogate_elbo_particle = 0
        log_r = None

        # compute elbo and surrogate elbo
        for name, site in model_trace.nodes.items():
            if site["type"] == "sample":
                elbo_particle = elbo_particle + torch_item(site["log_prob_sum"])
                surrogate_elbo_particle = surrogate_elbo_particle + site["log_prob_sum"]

        for name, site in guide_trace.nodes.items():
            if site["type"] == "sample":
                log_prob, score_function_term, entropy_term = site["score_parts"]

                elbo_particle = elbo_particle - torch_item(site["log_prob_sum"])

                # only change
                # if not is_identically_zero(entropy_term):
                #     surrogate_elbo_particle = surrogate_elbo_particle - entropy_term.sum()

                if not is_identically_zero(score_function_term):
                    if log_r is None:
                        log_r = _compute_log_r(model_trace, guide_trace)
                    site = log_r.sum_to(site["cond_indep_stack"])
                    surrogate_elbo_particle = surrogate_elbo_particle + (site * score_function_term).sum()

        return -elbo_particle, -surrogate_elbo_particle