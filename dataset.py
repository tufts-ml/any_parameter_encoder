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

def get_true_topics(n_topics, vocab_size):
    betas = get_toy_bar_betas(n_topics, vocab_size)
    topics = generate_topics(betas=betas, seed=0, shuffle=True)
    return topics

def generate_documents(topics, n_topics, vocab_size, avg_num_words, alpha=.05, seed=0):
    num_docs = 50
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

def create_toy_bar_docs(doc_file, n_topics, vocab_size):
    true_topics = get_true_topics(n_topics, vocab_size)
    docs, _ = generate_documents(true_topics, n_topics, vocab_size, 50)
    np.save(doc_file, docs)


class ToyBarsDataset(data.Dataset):
    def __init__(self, doc_file, topics_file, n_topics, vocab_size, alpha, use_cuda, training=True, generate=True):
        if not os.path.exists(doc_file):
            create_toy_bar_docs(doc_file, n_topics, vocab_size)
        self.documents = np.load(doc_file)
        self.topics = np.load(os.path.join(topics_file))
        self.num_docs = len(self.documents)
        self.num_models = len(self.topics)
        self.n_topics = n_topics
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.use_cuda = use_cuda
        self.training = training
        self.generate = generate

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
            topics = self.topics[index % self.num_topics]
        document = self.documents[index % self.num_docs]
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        document = torch.from_numpy(document.astype(np.float32)).type(dtype)
        topics = torch.from_numpy(topics.astype(np.float32)).type(dtype)
        return document, topics