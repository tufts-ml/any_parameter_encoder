import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import argparse
import itertools

from visualization.reconstructions import plot_side_by_side_docs
from datasets.load import load_toy_bars

parser = argparse.ArgumentParser(description='Results summary')
parser.add_argument('model_path', type=str, help='path to model')
args = parser.parse_args()

results_dir = os.path.dirname(args.model_path)
file_name = os.path.basename(args.model_path)
n_hidden_layers = int(file_name.rsplit('_', 2)[1])
h5f = h5py.File(os.path.join(results_dir, file_name), 'r')
weights_per_layer = []
biases_per_layer = []
for i in range(n_hidden_layers):
    weights_per_layer.append(h5f['weights_{}'.format(i + 1)][()])
    biases_per_layer.append(h5f['biases_{}'.format(i + 1)][()])

# print(np.array(weights_per_layer).max())
# print(np.array(weights_per_layer).min())
# print(np.array(biases_per_layer).max())
# print(np.array(biases_per_layer).min())

# else:
#             for weight_col_subset in weight_col.reshape(-1, 10, 10):
#             ax.imshow(weight_col_subset, cmap='PiYG', vmin=-2, vmax=2)
# if len(weight_col) == 100:
for i, weights in enumerate(weights_per_layer):
    if weights.shape == (100, 100):
        fig, axes = plt.subplots(10, 10)
        for j, weight_col in enumerate(weights.T):
            ax = axes[j/10, j%10]
            ax.imshow(weight_col.reshape(10, 10), cmap='PiYG', vmin=-2, vmax=2)
            ax.axis('off')
        plt.savefig(os.path.join(results_dir, 'weights_per_layer_{}.png'.format(str(i).zfill(3))), bbox_inches='tight')
    else:
        for k, weights_subset in enumerate(weights.reshape(-1, 100, 100)):
            fig, axes = plt.subplots(10, 10)
            for j, weight_col in enumerate(weights_subset.T):
                ax = axes[j/10, j%10]
                ax.imshow(weight_col.reshape(10, 10), cmap='PiYG', vmin=-2, vmax=2)
                ax.axis('off')
            plt.savefig(os.path.join(results_dir, 'weights_per_layer_{}_{}.png'.format(str(i).zfill(3), str(k).zfill(3))), bbox_inches='tight')

# for i, biases in enumerate(biases_per_layer):
#     plt.imshow(biases.reshape(10, 10), cmap='PiYG', vmin=-2, vmax=2)
#     plt.savefig(os.path.join(results_dir, 'biases_per_layer_{}.png'.format(i)))

# sample_dir = os.path.join(results_dir, 'samples')
# if not os.path.exists(sample_dir):
#     os.mkdir(sample_dir)
# dataset_names = ['train', 'valid', 'test', 'test_single', 'test_double', 'test_triple']
# vocab_size = 100
# datasets = load_toy_bars('toy_bars_10x10', VOCAB_SIZE=vocab_size)

# try:
#     train_topics = np.load(os.path.join(results_dir, 'train_topics.npy'))
#     documents = np.load(os.path.join(results_dir, 'documents.npy'))
# except:
#     train_topics = np.load(os.path.join('experiments', 'train_topics.npy'))
#     documents = np.load(os.path.join('experiments', 'documents.npy'))
# datasets = list(itertools.product(documents, train_topics))


# def plot_activations(dataset_idx, datapoint_idx):
#     activation = datasets[dataset_idx][datapoint_idx]
#     doc_and_activations = []
#     doc_and_activations.append(activation)
#     for weights, biases in zip(weights_per_layer, biases_per_layer):
#         activation = np.dot(weights, activation) + biases
#         doc_and_activations.append(activation)
#     fig, axes = plt.subplots(2, 3)
#     for i, activation in enumerate(doc_and_activations):
#         min_act = activation.min()
#         max_act = activation.max()
#         boundary = max(-min_act, max_act)
#         if activation.shape == (100,):
#             activation = activation.reshape(10, 10)
#         ax = axes[i / 3, i % 3]
#         ax.imshow(activation, cmap='PiYG', vmin=-boundary, vmax=boundary)
#         ax.set_title(round(boundary, 3))
#     plt.savefig(os.path.join(sample_dir, '{}_{}.png'.format(dataset_idx, datapoint_idx)), bbox_inches='tight')
#     np.save(os.path.join(sample_dir, '{}_{}.npy'.format(dataset_idx, datapoint_idx)), np.array(doc_and_activations))

# plot_activations(1, 0)
# plot_activations(3, 0)
# plot_activations(4, 0)
# plot_activations(5, 0)
