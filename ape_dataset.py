import os
import numpy as np
import torch
from torch.utils import data

from dataset import create_toy_bar_docs, generate_topics, generate_documents


def get_symmetric_betas(beta, n_topics, vocab_size):
    return beta * np.ones((n_topics, vocab_size))

def create_docs(doc_file, n_topics, vocab_size, num_docs):
    if 'same' in doc_file:
        create_toy_bar_docs(doc_file, n_topics, vocab_size, num_docs)
    elif 'sim' in doc_file:
        create_toy_bar_docs(doc_file, n_topics, vocab_size, num_docs, seed=1)
    elif 'diff' in doc_file:
        betas = get_symmetric_betas(1, n_topics, vocab_size)
        topics = generate_topics(betas, seed=0)
        docs, _ = generate_documents(topics, n_topics, vocab_size, avg_num_words=50, alpha=.1, seed=0, num_docs=num_docs)
        np.save(doc_file, docs)
    else:
        raise ValueError('doc_file must include "same," "sim," or "diff"')

def generate_topic_sets(topics_file, num_models, betas, seed=1):
    topic_sets = np.array([generate_topics(betas, seed) for _ in range(num_models)])
    np.save(topics_file, topic_sets)

def create_topics(topics_file, n_topics, vocab_size, num_models):
    if 'same' in topics_file:
        betas = get_symmetric_betas(.1, n_topics, vocab_size)
        generate_topic_sets(topics_file, num_models, betas, seed=0)
    elif 'sim' in topics_file:
        betas = get_symmetric_betas(.1, n_topics, vocab_size)
        generate_topic_sets(topics_file, num_models, betas, seed=1)
    elif 'diff' in topics_file:
        betas = get_symmetric_betas(.01, n_topics, vocab_size)
        generate_topic_sets(topics_file, num_models, betas, seed=0)
    else:
        raise ValueError('doc_file must include "same," "sim," or "diff"')

class APEDataset(data.Dataset):
    def __init__(self, doc_file, topics_file, n_topics, vocab_size, use_cuda, num_models, num_docs):
        if not os.path.exists(doc_file):
            create_docs(doc_file, n_topics, vocab_size, num_docs)
        if not os.path.exists(topics_file):
            create_topics(topics_file, n_topics, vocab_size, num_models)
        device = torch.device("cuda:0" if use_cuda else "cpu")
        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        documents = np.load(doc_file)
        topics = np.load(topics_file)

        if num_docs > len(documents):
            raise ValueError(f'{doc_file} contains less than {num_docs} documents')
        elif num_docs < len(documents):
            documents = documents[:num_docs]

        if num_models > len(topics):
            raise ValueError(f'{topics_file} contains less than {num_models} topics')
        elif num_models < len(topics):
            documents = documents[:num_models]

        self.documents = torch.from_numpy(documents).to(device).type(dtype)
        self.topics = torch.from_numpy(topics).to(device).type(dtype)
        self.num_docs = num_docs
        self.num_models = num_models

    def __len__(self):
        """ Denotes the total number of samples """
        return self.num_models * self.num_docs

    def __getitem__(self, index):
        """ Generates one sample of data """
        topics = self.topics[index % self.num_models]
        document = self.documents[index % self.num_docs]
        return document, topics
