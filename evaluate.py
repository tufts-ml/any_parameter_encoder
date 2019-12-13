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
    def __init__(self, encoder, *args, **kwargs):
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
        # we do not include tracelist generation in the time
        # since the posterior is already fully specified before then
        start = time.time()
        posterior = super(MCMC, self).run(*args, **kwargs)
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