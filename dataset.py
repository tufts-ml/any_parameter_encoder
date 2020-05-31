import os

import math
import numpy as np
import torch
from torch.utils import data

from utils import normalize1d


torch.manual_seed(0)

def generate_topics(betas, seed, shuffle=True):
    """ Generate a single set of topics """
    np.random.seed(seed)
    if shuffle:
        np.random.shuffle(betas)
    topics = []
    for beta in betas:
        topics.append(np.random.dirichlet(beta))
    return np.array(topics)

def get_toy_bar_betas(n_topics, vocab_size):
    """ Very strong beta prior which biases towards topics that look like bars """
    betas = []
    for i in range(n_topics):
        beta = np.ones(vocab_size)
        dim = math.sqrt(vocab_size)
        if i < dim:
            popular_words = [idx for idx in range(vocab_size) if idx % dim == i]
        else:
            popular_words = [idx for idx in range(vocab_size) if int(idx / dim) == i - dim]
        beta[popular_words] = 1000
        beta = normalize1d(beta)
        beta[popular_words] *= 5
        betas.append(beta)
    return betas

def get_true_topics(n_topics, vocab_size, topics_file):
    betas = get_toy_bar_betas(n_topics, vocab_size)
    topics = generate_topics(betas=betas, seed=0, shuffle=False)
    np.save(topics_file, np.expand_dims(topics, 0))
    return topics

def generate_documents(topics, n_topics, vocab_size, avg_num_words, alpha=.05, seed=0, num_docs=50):
    np.random.seed(seed)
    doc_topic_dists = np.random.dirichlet([alpha] * n_topics, size=num_docs)
    documents = []
    for pi in doc_topic_dists:
        num_words = np.random.poisson(avg_num_words)
        
        doc = np.zeros(vocab_size)
        for _ in range(num_words):
            z = np.random.choice(range(n_topics), p=pi)
            doc += np.random.multinomial(1, topics[z])
        documents.append(doc.astype(np.float32))
    return documents, doc_topic_dists

def create_toy_bar_docs(doc_file, n_topics, vocab_size, num_docs=50, seed=0, avg_num_words=50):
    true_topics = get_true_topics(n_topics, vocab_size, topics_file='true_topics.npy')
    docs, true_dist = generate_documents(true_topics, n_topics, vocab_size, avg_num_words=avg_num_words, num_docs=num_docs, seed=seed)
    np.save(doc_file, docs)
    np.save(doc_file.replace('.npy', '_dist.npy'), true_dist)


class ToyBarsDataset(data.Dataset):
    def __init__(self, doc_file, n_topics, vocab_size, alpha, use_cuda, topics_file=None, num_models=None, training=True, generate=True, subset_docs=None, avg_num_words=50, num_docs=50):
        if not os.path.exists(topics_file):
            print('Creating', topics_file)
            topics = get_true_topics(n_topics, vocab_size, topics_file)
        else:
            topics = np.load(topics_file)[0]
        if not os.path.exists(doc_file):
            print('Creating ', doc_file)
            docs, true_dist = generate_documents(topics, n_topics, vocab_size, avg_num_words=avg_num_words, num_docs=num_docs, seed=0)
            np.save(doc_file, docs)
            np.save(doc_file.replace('.npy', '_dist.npy'), true_dist)
        device = torch.device("cuda:0" if use_cuda else "cpu")
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.documents = torch.from_numpy(np.load(doc_file)).type(dtype)
        if subset_docs:
            self.documents = self.documents[:subset_docs]
        self.num_docs = len(self.documents)
        self.n_topics = n_topics
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.use_cuda = use_cuda
        self.training = training
        self.generate = generate
        if generate:
            self.num_models = num_models
        else:
            self.topics = torch.from_numpy(np.load(os.path.join(topics_file)))
            self.topics = self.topics.to(device).type(dtype)
            self.num_models = len(self.topics)
        self.documents = self.documents.to(device)
        

    def __len__(self):
        """ Denotes the total number of samples """
        return self.num_models * self.num_docs

    def __getitem__(self, index):
        """ Generates one sample of data """
        if self.generate:
            if self.training:
                seed = index
            else:
                seed = index + self.num_models * self.num_docs
            np.random.seed(seed)
            topics = np.random.dirichlet(np.ones(self.vocab_size), size=self.n_topics)
        else:
            topics = self.topics[index % self.num_models]
        document = self.documents[index % self.num_docs]
        # dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        # document = torch.from_numpy(document.astype(np.float32)).type(dtype)
        # topics = torch.from_numpy(topics.astype(np.float32)).type(dtype)
        return document, topics


class NonToyBarsDataset(ToyBarsDataset):
    def __init__(self, doc_file, n_topics, vocab_size, alpha, use_cuda, topics_file=None, num_models=None, training=True, generate=True, subset_docs=None, avg_num_words=50, num_docs=500):
        if not os.path.exists(topics_file):
            topics = generate_topics(np.ones((n_topics, vocab_size)) * .1, seed=0)
            np.save(topics_file, np.expand_dims(topics, 0))
        if not os.path.exists(doc_file):
            print('Creating ', doc_file)
            topics = np.load(topics_file)[0]
            docs, true_dist = generate_documents(topics, n_topics, vocab_size, avg_num_words=avg_num_words, num_docs=num_docs, seed=0)
            np.save(doc_file, docs)
            np.save(doc_file.replace('.npy', '_dist.npy'), true_dist)
        device = torch.device("cuda:0" if use_cuda else "cpu")
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.documents = torch.from_numpy(np.load(doc_file)).type(dtype)
        if subset_docs:
            self.documents = self.documents[:subset_docs]
        self.num_docs = len(self.documents)
        self.n_topics = n_topics
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.use_cuda = use_cuda
        self.training = training
        self.generate = generate
        if generate:
            self.num_models = num_models
        else:
            self.topics = torch.from_numpy(np.load(os.path.join(topics_file)))
            self.topics = self.topics.to(device).type(dtype)
            self.num_models = len(self.topics)
        self.documents = self.documents.to(device)
        


class ToyBarsDocsDataset(data.Dataset):
    def __init__(self, doc_file, n_topics, vocab_size, alpha, use_cuda, training=True, generate=True, subset_docs=None, avg_num_words=50):
        if not os.path.exists(doc_file):
            print('Creating ', doc_file)
            create_toy_bar_docs(doc_file, n_topics, vocab_size, num_docs=100000, avg_num_words=avg_num_words)
        device = torch.device("cuda:0" if use_cuda else "cpu")
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.documents = torch.from_numpy(np.load(doc_file)).type(dtype)
        if subset_docs:
            self.documents = self.documents[:subset_docs]
        self.num_docs = len(self.documents)
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.use_cuda = use_cuda
        self.training = training
        self.generate = generate
        self.documents = self.documents.to(device)
        

    def __len__(self):
        """ Denotes the total number of samples """
        if self.training:
            return int(self.num_docs * .8)
        else:
            return int(self.num_docs * .2)

    def __getitem__(self, index):
        """ Generates one sample of data """
        num_train = self.num_docs * .8
        if self.training:
            idx = int(index % num_train)
        else:
            idx = int(index + num_train)

        document = self.documents[idx]
        # dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        # document = torch.from_numpy(document.astype(np.float32)).type(dtype)
        # topics = torch.from_numpy(topics.astype(np.float32)).type(dtype)
        return document