import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

from visualization.reconstructions import plot_side_by_side_docs
from datasets.load import load_toy_bars


results_dir = 'experiments/vae_experiments/10x10_full9'
file_name = 'lda_orig_100000_samples_5_100.h5'
n_hidden_layers = 5
h5f = h5py.File(os.path.join(results_dir, file_name), 'r')
weights_per_layer = []
biases_per_layer = []
for i in range(n_hidden_layers):
    weights_per_layer.append(h5f['weights_{}'.format(i + 1)][()])
    biases_per_layer.append(h5f['biases_{}'.format(i + 1)][()])

print(np.array(weights_per_layer).max())
print(np.array(weights_per_layer).min())
print(np.array(biases_per_layer).max())
print(np.array(biases_per_layer).min())

# for i, weights in enumerate(weights_per_layer):
#     fig, axes = plt.subplots(10, 10)
#     for j, weight_col in enumerate(weights.T):
#         ax = axes[j/10, j%10]
#         ax.imshow(weight_col.reshape(10, 10), cmap='PiYG', vmin=-2, vmax=2)
#         ax.axis('off')
#     plt.savefig(os.path.join(results_dir, 'weights_per_layer_{}.png'.format(i)), bbox_inches='tight')

# for i, biases in enumerate(biases_per_layer):
#     plt.imshow(biases.reshape(10, 10), cmap='PiYG', vmin=-2, vmax=2)
#     plt.savefig(os.path.join(results_dir, 'biases_per_layer_{}.png'.format(i)))

sample_dir = os.path.join(results_dir, 'samples')
if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)
dataset_names = ['train', 'valid', 'test', 'test_single', 'test_double', 'test_triple']
vocab_size = 100
datasets = load_toy_bars('toy_bars_10x10', VOCAB_SIZE=vocab_size)
# dataset_idx = 0
# datapoint_idx = 0
def plot_activations(dataset_idx, datapoint_idx):
    activation = datasets[dataset_idx][datapoint_idx]
    doc_and_activations = []
    doc_and_activations.append(activation)
    for weights, biases in zip(weights_per_layer, biases_per_layer):
        activation = np.dot(weights, activation) + biases
        doc_and_activations.append(activation)
    fig, axes = plt.subplots(2, 3)
    for i, activation in enumerate(doc_and_activations):
        min_act = activation.min()
        max_act = activation.max()
        boundary = max(-min_act, max_act)
        if activation.shape == (100,):
            activation = activation.reshape(10, 10)
        ax = axes[i / 3, i % 3]
        ax.imshow(activation, cmap='PiYG', vmin=-boundary, vmax=boundary)
        ax.set_title(round(boundary, 3))
    plt.savefig(os.path.join(sample_dir, '{}_{}.png'.format(dataset_idx, datapoint_idx)), bbox_inches='tight')
    np.save(os.path.join(sample_dir, '{}_{}.npy'.format(dataset_idx, datapoint_idx)), np.array(doc_and_activations))

plot_activations(1, 0)
plot_activations(3, 0)
plot_activations(4, 0)
plot_activations(5, 0)

        # state_dict['encoder.fcmu.weight'] = torch.from_numpy(h5f['weights_out_mean'][()]).t()
        # state_dict['encoder.fcsigma.weight'] = torch.from_numpy(h5f['weights_out_log_sigma'][()]).t()
        # state_dict['encoder.fcmu.bias'] = torch.from_numpy(h5f['biases_out_mean'][()])
        # state_dict['encoder.fcsigma.bias'] = torch.from_numpy(h5f['biases_out_log_sigma'][()])

        # state_dict['encoder.bnmu.bias'] = torch.from_numpy(h5f['beta_out_mean'][()])
        # state_dict['encoder.bnsigma.bias'] = torch.from_numpy(h5f['beta_out_log_sigma'][()])
        # state_dict['encoder.bnmu.weight'] = torch.ones(self.n_topics)
        # state_dict['encoder.bnsigma.weight'] = torch.ones(self.n_topics)
        # state_dict['encoder.bnmu.running_mean'] = torch.from_numpy(h5f['running_mean_out_mean'][()])
        # state_dict['encoder.bnsigma.running_mean'] = torch.from_numpy(h5f['running_mean_out_log_sigma'][()])
        # state_dict['encoder.bnmu.running_var'] = torch.from_numpy(h5f['running_var_out_mean'][()])
        # state_dict['encoder.bnsigma.running_var'] = torch.from_numpy(h5f['running_var_out_log_sigma'][()])

        # state_dict['encoder.scale'] = torch.from_numpy(np.array(h5f['scale'][()]))

        # if not self.topic_init:
            # state_dict['decoder.topics'] = torch.from_numpy(h5f['topics'][()])
