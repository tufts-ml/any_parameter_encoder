import os

from visualization.reconstructions import plot_saved_samples
from datasets.load import load_toy_bars


datadir = 'toy_bars_10x10'
results_dir = 'problem_toy_bars'
vocab_size = 100
sample_idx = list(range(10))
datasets = load_toy_bars(datadir)
dataset_names = ['train', 'valid', 'test', 'test_single', 'test_double', 'test_triple']

# n_hidden_units = 20
# n_hidden_layers = 2
# models = ['lda_orig', 'lda_scale', 'lda_orig_hallucinations', 'lda_scale_hallucinations']
#
# for data_name, data in zip(dataset_names, datasets):
#     filenames = []
#     for model in models:
#         file = '_'.join(['vae', model, data_name, str(n_hidden_layers), str(n_hidden_units)]) + '.npy'
#         filepath = os.path.join(os.getcwd(), results_dir, file)
#         filenames.append(filepath)
#
#     plot_name = data_name + '_reconstructions.pdf'
#     plot_saved_samples(data[sample_idx], filenames, plot_name, vocab_size=vocab_size)

# for data_name, data in zip(dataset_names, datasets):
#     filenames = [os.path.join(os.getcwd(), results_dir, 'vae_lda_scale_{}_5_20.npy'.format(data_name))  ]
#     plot_name = os.path.join(results_dir, '{}_scale_5_20.pdf'.format(data_name))
#     plot_saved_samples(data[sample_idx], filenames, plot_name, vocab_size=vocab_size)


# for data_name, data in zip(dataset_names, datasets):
#     filenames = [os.path.join(os.getcwd(), results_dir, 'svi_lda_orig_{}_1_10.npy'.format(data_name)),
#                  os.path.join(os.getcwd(), results_dir, 'mcmc_lda_orig_{}_1_10.npy'.format(data_name))]
#     plot_name = os.path.join(results_dir, '{}_svi_vs_mcmc.pdf'.format(data_name))
#     plot_saved_samples(data[sample_idx], filenames, plot_name, vocab_size=vocab_size)

# for data_name, data in zip(dataset_names, datasets):
#     if data_name == 'train':
#         n_hidden_layers = 0
#     elif data_name == 'valid':
#         n_hidden_layers = 3
#     elif data_name == 'test':
#         n_hidden_layers = 3
#     elif data_name == 'test_single':
#         n_hidden_layers = 3
#     elif data_name == 'test_double':
#         n_hidden_layers = 4
#     elif data_name == 'test_triple':
#         n_hidden_layers = 4
#     filenames = [os.path.join(os.getcwd(), 'svi_baseline', 'svi_lda_orig_{}_{}_10.npy'.format(data_name, n_hidden_layers)),
#                  os.path.join(os.getcwd(), results_dir, 'mcmc_lda_orig_{}_1_10.npy'.format(data_name))]
#     plot_name = os.path.join('svi_baseline', '{}_best_svi_vs_mcmc.pdf'.format(data_name))
#     plot_saved_samples(data[sample_idx], filenames, plot_name, vocab_size=vocab_size)


filenames = [os.path.join(os.getcwd(), results_dir, 'vae_lda_scale_valid_5_20.npy'),
             os.path.join(os.getcwd(), results_dir, 'svi_lda_orig_valid_1_10.npy'),
             os.path.join(os.getcwd(), results_dir, 'mcmc_lda_orig_valid_1_10.npy'),]
plot_name = os.path.join(results_dir, 'valid_compare_inference.pdf')
plot_saved_samples(datasets[1][sample_idx], filenames, plot_name, vocab_size=vocab_size)