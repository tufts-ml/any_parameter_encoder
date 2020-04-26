import os
import glob
import numpy as np
import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist

import logging

class MLP(nn.Module):
    def __init__(self, n_input_units, n_output_units):
        super(MLP, self).__init__()
        self.fc = nn.Linear(n_input_units, n_output_units)
        self.relu = F.relu
        self.bn = nn.BatchNorm1d(n_output_units)

    def forward(self, x):
        return self.bn(self.relu(self.fc(x)))

class MLP_with_skip(MLP):
    def forward(self, x):
        return self.bn(x + self.relu(self.fc(x)))

class Encoder(nn.Module):
    def __init__(self, n_hidden_units, n_hidden_layers, architecture, n_topics=4, vocab_size=9, use_scale=False, skip_connections=False,
                 model_type='avitm'):
        super(Encoder, self).__init__()
        self.n_hidden_layers = n_hidden_layers
        self.vocab_size = vocab_size
        self.n_topics = n_topics
        # setup the non-linearities
        self.relu = F.relu
        self.skip_connections = skip_connections
        self.model_type = model_type
        # encoder Linear layers
        modules = []
        self.architecture = architecture
        if architecture == 'naive' or architecture == 'naive_separated':
            modules.append(MLP((1 + n_topics) * vocab_size, n_hidden_units))
        elif architecture in ['template', 'template_unnorm', 'pseudo_inverse', 'pseudo_inverse_scaled']:
            modules.append(MLP(n_topics, n_hidden_units))
        elif architecture == 'template_plus_topics':
            modules.append(MLP(n_topics * (1 + vocab_size), n_hidden_units))
        elif architecture == 'standard':
            modules.append(MLP(vocab_size, n_hidden_units))
        else:
            raise ValueError('Invalid architecture')
        for i in range(self.n_hidden_layers - 1):
            if self.skip_connections and i % 2 == 0:
                modules.append(MLP_with_skip(n_hidden_units, n_hidden_units))
            else:
                modules.append(MLP(n_hidden_units, n_hidden_units))

        if architecture == 'naive_separated':
            self.enc_layers_mu = nn.Sequential(*modules)
            self.enc_layers_sigma = nn.Sequential(*modules)
        else:
            self.enc_layers = nn.Sequential(*modules)
        self.fcmu = nn.Linear(n_hidden_units, n_topics)
        self.fcsigma = nn.Linear(n_hidden_units, n_topics)

        if self.n_hidden_layers == 0 and n_hidden_units == n_topics:
            self.fcmu.weight.data = torch.eye(n_topics).data
            self.fcsigma.weight.data = torch.zeros((n_hidden_units, n_topics)).data
            self.fcmu.bias.data = torch.zeros(n_topics).data
            self.fcsigma.bias.data = torch.ones(n_topics).data

        self.bnmu = nn.BatchNorm1d(n_topics)
        self.bnsigma = nn.BatchNorm1d(n_topics)
        self.use_scale = use_scale
        if self.use_scale:
            self.scale = nn.Parameter(torch.ones([]))

    def forward(self, x, topics):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.reshape(-1, self.vocab_size)
        if self.architecture == 'naive':
            x_and_topics = torch.cat((x, topics.reshape(-1, self.n_topics * self.vocab_size)), dim=1)
            # then return a mean vector and a (positive) square root covariance
            # each of size batch_size x n_topics
            z_loc = self.bnmu(self.fcmu(self.enc_layers(x_and_topics)))
            z_scale = torch.sqrt(torch.exp(self.bnsigma(self.fcsigma(self.enc_layers(x_and_topics)))))
        elif self.architecture == 'template':
            x_and_topics = torch.einsum("ab,abc->ac", (x, torch.transpose(topics, 1, 2)))
            x_and_topics = torch.div(x_and_topics, torch.sum(x_and_topics, dim=1).reshape((-1, 1)))
            z_loc = self.bnmu(self.fcmu(self.enc_layers(x_and_topics)))
            z_scale = torch.sqrt(torch.exp(self.bnsigma(self.fcsigma(self.enc_layers(x_and_topics)))))
        elif self.architecture == 'template_plus_topics':
            x_and_topics = torch.einsum("ab,abc->ac", (x, torch.transpose(topics, 1, 2)))
            x_and_topics = torch.div(x_and_topics, torch.sum(x_and_topics, dim=1).reshape((-1, 1)))
            x_and_topics = torch.cat((x_and_topics, topics.reshape(-1, self.n_topics * self.vocab_size)), dim=1)
            z_loc = self.bnmu(self.fcmu(self.enc_layers(x_and_topics)))
            z_scale = torch.sqrt(torch.exp(self.bnsigma(self.fcsigma(self.enc_layers(x_and_topics)))))
        elif self.architecture == 'standard':
            z_loc = self.bnmu(self.fcmu(self.enc_layers(x)))
            z_scale = torch.sqrt(torch.exp(self.bnsigma(self.fcsigma(self.enc_layers(x)))))
        elif self.architecture == 'naive_separated':
            x_and_topics = torch.cat((x, topics.reshape(-1, self.n_topics * self.vocab_size)), dim=1)
            # then return a mean vector and a (positive) square root covariance
            # each of size batch_size x n_topics
            z_loc = self.bnmu(self.fcmu(self.enc_layers_mu(x_and_topics)))
            z_scale = torch.sqrt(torch.exp(self.bnsigma(self.fcsigma(self.enc_layers_sigma(x_and_topics)))))
        elif self.architecture == 'template_unnorm':
            x_and_topics = torch.einsum("ab,abc->ac", (x, torch.transpose(topics, 1, 2)))
            z_loc = self.bnmu(self.fcmu(self.enc_layers(x_and_topics)))
            z_scale = torch.sqrt(torch.exp(self.bnsigma(self.fcsigma(self.enc_layers(x_and_topics)))))
        elif self.architecture == 'pseudo_inverse':
            x_and_topics = torch.einsum("ab,abc->ac", (x, torch.transpose(topics, 1, 2)))  # [batch, n_topics] 
            topics_topics_t = torch.einsum("abc,acd->abd", (topics, torch.transpose(topics, 1, 2)))  # [batch, n_topics, n_topics]
            x_and_topics = torch.einsum("abd,ad->ab", (topics_topics_t, x_and_topics))  # [batch, n_topics]
            z_loc = self.bnmu(self.fcmu(self.enc_layers(x_and_topics)))
            z_scale = torch.sqrt(torch.exp(self.bnsigma(self.fcsigma(self.enc_layers(x_and_topics)))))
        elif self.architecture == 'pseudo_inverse_scaled':
            x = torch.div(x, torch.sum(x, dim=1).reshape((-1, 1)))
            if self.model_type == 'nvdm':
                # log(1 + x) to avoid nans
                x = torch.log1p(x)
            x_and_topics = torch.einsum("ab,abc->ac", (x, torch.transpose(topics, 1, 2)))  # [batch, n_topics] 
            topics_topics_t = torch.einsum("abc,acd->abd", (topics, torch.transpose(topics, 1, 2)))  # [batch, n_topics, n_topics]
            x_and_topics = torch.einsum("abd,ad->ab", (topics_topics_t, x_and_topics))  # [batch, n_topics]
            if self.model_type == 'avitm':
                x_and_topics = torch.log(x_and_topics)
            z_loc = self.bnmu(self.fcmu(self.enc_layers(x_and_topics)))
            z_scale = torch.sqrt(torch.exp(self.bnsigma(self.fcsigma(self.enc_layers(x_and_topics)))))
        else:
            raise ValueError('Invalid architecture')
        if self.use_scale:
            z_loc = torch.mul(self.scale, z_loc)
        return z_loc, z_scale


class Decoder(nn.Module):
    def __init__(self, use_scale=False, model_type='avitm'):
        super(Decoder, self).__init__()
        self.use_scale = use_scale
        if self.use_scale:
            self.scale = nn.Parameter(torch.ones([]), requires_grad=False)
        self.softmax = nn.Softmax(dim=1)
        self.model_type = model_type

    def forward(self, z, topics):
        if self.use_scale:
            z = torch.mul(autograd.Variable(self.scale), z)
        if self.model_type == 'avitm':
            word_probs = torch.matmul(self.softmax(z), topics)
        elif self.model_type == 'nvdm':
            bias = torch.zeros(topics.size()[1], requires_grad=True)
            word_probs = self.softmax(torch.matmul(z, topics).add(bias))
        else:
            raise ValueError('Passed unsupported model_type: ', self.model_type)
        return word_probs


class APE(nn.Module):
    def __init__(self, n_hidden_units=100, n_hidden_layers=2, results_dir=None,
                 alpha=.1, vocab_size=9, n_topics=4, use_cuda=False, architecture='naive', 
                 scale_type='sample', skip_connections=False, model_type='avitm', **kwargs):
        super(APE, self).__init__()

        # create the encoder and decoder networks
        self.scale_type = scale_type
        if self.scale_type == 'sample':
            self.encoder = Encoder(
                n_hidden_units,
                n_hidden_layers,
                architecture,
                n_topics=n_topics,
                vocab_size=vocab_size,
                use_scale=False,
                skip_connections=skip_connections,
                model_type=model_type)
            self.decoder = Decoder(use_scale=True)
        elif self.scale_type == 'mean':
            self.encoder = Encoder(
                n_hidden_units,
                n_hidden_layers,
                architecture,
                n_topics=n_topics,
                vocab_size=vocab_size,
                use_scale=True,
                skip_connections=skip_connections,
                model_type=model_type)
            self.decoder = Decoder(use_scale=False, model_type=model_type)
        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
            device = 'cuda:0'
        else:
            device = 'cpu'
        self.use_cuda = use_cuda
        self.n_topics = n_topics
        alpha_vec = alpha * np.ones((1, n_topics)).astype(np.float32)
        self.z_loc = torch.from_numpy((np.log(alpha_vec).T - np.mean(np.log(alpha_vec), 1)).T).float().to(device)
        self.z_scale = torch.from_numpy((
            ((1.0 / alpha_vec) * (1 - (2.0 / n_topics))).T +
            (1.0 / (n_topics * n_topics)) * np.sum(1.0 / alpha_vec, 1)
        ).T).float().to(device)

        self.n_topics = n_topics
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units
        self.results_dir = results_dir
        self.architecture = architecture

    def model(self, x, topics):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z = pyro.sample("latent",
                            dist.Normal(self.z_loc, self.z_scale).to_event(1))
            word_probs = self.decoder.forward(z, topics)
            return pyro.sample("doc_words",
                        dist.Multinomial(probs=word_probs),
                        obs=x)

    def encoder_guide(self, x, topics):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x, topics)
            # sample the latent code z
            pyro.sample("latent",
                        dist.Normal(z_loc, z_scale).to_event(1))

    def mean_field_guide(self, x, topics):
        with pyro.plate("data", x.shape[0]):
            z_loc = pyro.param("z_loc", x.new_zeros(torch.Size((x.shape[0], self.n_topics))))
            z_scale = pyro.param("z_scale", x.new_ones(torch.Size((x.shape[0], self.n_topics))))

            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def map_guide(self, x, topics):
        with pyro.plate("data", x.shape[0]):
            z_loc = pyro.param("z_loc", self.z_loc)
            pyro.sample("latent", dist.Delta(z_loc).to_event(1))


class Encoder_APE_VAE(Encoder):
    """
    TODO: instead of subclassing just if if/else branching.
    The only difference is one less dimension for the topics (e.g. no batch dim)
    """
    def __init__(self, n_hidden_units, n_hidden_layers, architecture, **kwargs):
        super(Encoder_APE_VAE, self).__init__(n_hidden_units, n_hidden_layers, architecture, **kwargs)
    
    def forward(self, x, topics):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.reshape(-1, self.vocab_size)
        if self.architecture == 'naive':
            x_and_topics = torch.cat((x, topics.reshape(-1, self.n_topics * self.vocab_size)), dim=1)
            # then return a mean vector and a (positive) square root covariance
            # each of size batch_size x n_topics
            z_loc = self.bnmu(self.fcmu(self.enc_layers(x_and_topics)))
            z_scale = torch.sqrt(torch.exp(self.bnsigma(self.fcsigma(self.enc_layers(x_and_topics)))))
        elif self.architecture == 'template':
            x_and_topics = torch.matmul(x, torch.transpose(topics, 0, 1))
            x_and_topics = torch.div(x_and_topics, torch.sum(x_and_topics, dim=1).reshape((-1, 1)))
            z_loc = self.bnmu(self.fcmu(self.enc_layers(x_and_topics)))
            z_scale = torch.sqrt(torch.exp(self.bnsigma(self.fcsigma(self.enc_layers(x_and_topics)))))
        elif self.architecture == 'template_plus_topics':
            x_and_topics = torch.matmul(x, torch.transpose(topics, 0, 1))
            x_and_topics = torch.div(x_and_topics, torch.sum(x_and_topics, dim=1).reshape((-1, 1)))
            x_and_topics = torch.cat((x_and_topics, topics.reshape(-1, self.n_topics * self.vocab_size)), dim=1)
            z_loc = self.bnmu(self.fcmu(self.enc_layers(x_and_topics)))
            z_scale = torch.sqrt(torch.exp(self.bnsigma(self.fcsigma(self.enc_layers(x_and_topics)))))
        elif self.architecture == 'standard':
            z_loc = self.bnmu(self.fcmu(self.enc_layers(x)))
            z_scale = torch.sqrt(torch.exp(self.bnsigma(self.fcsigma(self.enc_layers(x)))))
        elif self.architecture == 'naive_separated':
            x_and_topics = torch.cat((x, topics.reshape(-1, self.n_topics * self.vocab_size)), dim=1)
            # then return a mean vector and a (positive) square root covariance
            # each of size batch_size x n_topics
            z_loc = self.bnmu(self.fcmu(self.enc_layers_mu(x_and_topics)))
            z_scale = torch.sqrt(torch.exp(self.bnsigma(self.fcsigma(self.enc_layers_sigma(x_and_topics)))))
        elif self.architecture == 'template_unnorm':
            x_and_topics = torch.einsum("ab,bc->ac", (x, torch.transpose(topics, 0, 1)))
            z_loc = self.bnmu(self.fcmu(self.enc_layers(x_and_topics)))
            z_scale = torch.sqrt(torch.exp(self.bnsigma(self.fcsigma(self.enc_layers(x_and_topics)))))
        elif self.architecture == 'pseudo_inverse':
            x_and_topics = torch.einsum("ab,bc->ac", (x, torch.transpose(topics, 0, 1)))  # [batch, n_topics]
            topics_topics_t = torch.einsum("bc,cb->bb", (topics, torch.transpose(topics, 0, 1)))  # [batch, n_topics, n_topics]
            x_and_topics = torch.einsum("bd,ad->ab", (topics_topics_t, x_and_topics))  # [batch, n_topics]
            z_loc = self.bnmu(self.fcmu(self.enc_layers(x_and_topics)))
            z_scale = torch.sqrt(torch.exp(self.bnsigma(self.fcsigma(self.enc_layers(x_and_topics)))))
        elif self.architecture == 'pseudo_inverse_scaled':
            x = torch.div(x, torch.sum(x, dim=1).reshape((-1, 1)))
            if self.model_type == 'nvdm':
                x = torch.log(x)
            x_and_topics = torch.einsum("ab,abc->ac", (x, torch.transpose(topics, 0, 1)))  # [batch, n_topics]
            topics_topics_t = torch.einsum("abc,acd->abd", (topics, torch.transpose(topics, 0, 1)))  # [batch, n_topics, n_topics]
            x_and_topics = torch.einsum("abd,ad->ab", (topics_topics_t, x_and_topics))  # [batch, n_topics]
            if self.model_type == 'avitm':
                x_and_topics = torch.log(x_and_topics)
            z_loc = self.bnmu(self.fcmu(self.enc_layers(x_and_topics)))
            z_scale = torch.sqrt(torch.exp(self.bnsigma(self.fcsigma(self.enc_layers(x_and_topics)))))
        else:
            raise ValueError('Invalid architecture')
        if self.use_scale:
            z_loc = torch.mul(self.scale, z_loc)
        return z_loc, z_scale


class APE_VAE(nn.Module):
    """
    The difference between APE and APE_VAE is that APE takes in topics during the training process, while APE_VAE learns topics via gradient descent.
    """
    def __init__(self, n_hidden_units=100, n_hidden_layers=2, results_dir=None,
                 alpha=.1, vocab_size=9, n_topics=4, use_cuda=False, architecture='naive',
                 scale_type='sample', skip_connections=False, model_type='avitm', **kwargs):
        super(APE_VAE, self).__init__()

        # create the encoder and decoder networks
        self.scale_type = scale_type
        if self.scale_type == 'sample':
            self.encoder = Encoder_APE_VAE(
                n_hidden_units,
                n_hidden_layers,
                architecture,
                n_topics=n_topics,
                vocab_size=vocab_size,
                use_scale=False,
                skip_connections=skip_connections)
            self.decoder = Decoder(use_scale=True)
        elif self.scale_type == 'mean':
            self.encoder = Encoder(
                n_hidden_units,
                n_hidden_layers,
                architecture,
                n_topics=n_topics,
                vocab_size=vocab_size,
                use_scale=True,
                skip_connections=skip_connections)
            self.decoder = Decoder(use_scale=False)
        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
            device = 'cuda:0'
        else:
            device = 'cpu'
        self.use_cuda = use_cuda
        self.n_topics = n_topics
        alpha_vec = alpha * np.ones((1, n_topics)).astype(np.float32)
        self.z_loc = torch.from_numpy((np.log(alpha_vec).T - np.mean(np.log(alpha_vec), 1)).T).float().to(device)
        self.z_scale = torch.from_numpy((
            ((1.0 / alpha_vec) * (1 - (2.0 / n_topics))).T +
            (1.0 / (n_topics * n_topics)) * np.sum(1.0 / alpha_vec, 1)
        ).T).float().to(device)

        self.n_topics = n_topics
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units
        self.results_dir = results_dir
        self.architecture = architecture
        self.topics = torch.empty(n_topics, vocab_size, requires_grad=True)
        torch.nn.init.xavier_normal_(self.topics, gain=1.0)
        self.topics = self.topics.to(device)

    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z = pyro.sample("latent",
                            dist.Normal(self.z_loc, self.z_scale).to_event(1))
            word_probs = self.decoder.forward(z, self.topics)
            return pyro.sample("doc_words",
                        dist.Multinomial(probs=word_probs),
                        obs=x)

    def encoder_guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x, self.topics)
            # sample the latent code z
            pyro.sample("latent",
                        dist.Normal(z_loc, z_scale).to_event(1))
