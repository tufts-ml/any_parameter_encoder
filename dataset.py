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
    topics = generate_topics(betas=betas, seed=0, shuffle=False)
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
    return np.array(documents), doc_topic_dists

def create_toy_bar_docs(
    doc_file, n_topics, vocab_size, num_docs=50, seed=0, avg_num_words=50, exact_toy_bars=False):
    if exact_toy_bars:
        true_topics = get_toy_bar_betas(n_topics, vocab_size)
        true_topics = [topics/sum(topics) for topics in true_topics]
    else:
        true_topics = get_true_topics(n_topics, vocab_size)
    docs, true_dist = generate_documents(true_topics, n_topics, vocab_size, avg_num_words=avg_num_words, num_docs=num_docs, seed=seed)
    np.save(doc_file, docs)
    np.save(doc_file.replace('.npy', '_dist.npy'), true_dist)


class ToyBarsDataset(data.Dataset):
    """
    Topics are generated based on alpha; documents are generated from toy bar-like topics.

    """
    def __init__(self, doc_file, topics_file, n_topics, vocab_size, alpha, use_cuda,
                 num_models, num_docs, avg_num_words=50, seed=0):
        if not os.path.exists(topics_file):
            print('Creating', topics_file)
            topics = []
            for i in range(num_models):
                topics.append(generate_topics(np.ones((n_topics, vocab_size)) * alpha, seed + i, shuffle=False))
            topics = np.array(topics)
            np.save(topics_file, topics)
        else:
            topics = np.load(topics_file)

        if not os.path.exists(doc_file):
            print('Creating ', doc_file)
            true_topics = get_true_topics(n_topics, vocab_size)
            documents, true_dist = generate_documents(
                true_topics, n_topics, vocab_size, avg_num_words=avg_num_words,
                num_docs=num_docs, seed=seed)
            np.save(doc_file, documents)
            np.save(doc_file.replace('.npy', '_dist.npy'), true_dist)
        else:
            documents = np.load(doc_file)

        if num_models < len(topics):
            topics = topics[:num_models]
        elif num_models > len(topics):
            raise ValueError(f'Already created {topics_file} with fewer topics than `num_models`.')

        if num_docs < len(documents):
            documents = documents[:num_docs]
        elif num_docs > len(documents):
            raise ValueError(f'Already created {doc_file} with fewer topics than `num_docs`.')

        self.num_models = num_models
        self.num_docs = num_docs
        device = torch.device("cuda:0" if use_cuda else "cpu")
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.documents = torch.from_numpy(documents).type(dtype).to(device)
        self.topics = torch.from_numpy(topics).type(dtype).to(device)
        
    def __len__(self):
        """ Denotes the total number of samples """
        return self.num_models * self.num_docs

    def __getitem__(self, index):
        """ Generates one sample of data """
        topics = self.topics[index % self.num_models]
        document = self.documents[index % self.num_docs]
        return document, topics


class ToyBarsDocsDataset(data.Dataset):
    """
    documents are generated from toy bar-like topics.

    """
    def __init__(self, doc_file, n_topics, vocab_size, alpha, use_cuda,
                 num_docs, avg_num_words=50, exact_toy_bars=False):
        if not os.path.exists(doc_file):
            print('Creating ', doc_file)
            create_toy_bar_docs(
                doc_file, n_topics, vocab_size, num_docs=num_docs,
                avg_num_words=avg_num_words, exact_toy_bars=exact_toy_bars
        )
        device = torch.device("cuda:0" if use_cuda else "cpu")
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.documents = torch.from_numpy(np.load(doc_file)).type(dtype).to(device)

        if num_docs < len(self.documents):
            self.documents = self.documents[:num_docs]
        elif num_docs > len(self.documents):
            raise ValueError(f'Already created {doc_file} with fewer topics than `num_docs`.')
        self.num_docs = num_docs

    def __len__(self):
        """ Denotes the total number of samples """
        return self.num_docs

    def __getitem__(self, index):
        """ Generates one sample of data """
        return self.documents[index % self.num_docs]