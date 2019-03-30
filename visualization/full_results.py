import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


results_dir = 'dump_fixed2'
results_csv = 'results.csv'

df = pd.read_csv(os.path.join(results_dir, results_csv), header=None)
df.columns = ['inference', 'model', 'dataset', 'n_hidden_layers', 'n_hidden_units', 'posterior_predictive_density']

# remove the 50 layer VAE results
df = df[df.n_hidden_layers != 50]

keys = ['SVI', 'MCMC', 'VAE (1 hidden layer)', 'VAE (2 hidden layers)', 'VAE (5 hidden layers)', 'VAE (10 hidden layers)',
        'VAE (20 hidden layers)'] #, 'VAE (50 hidden layers)']
colors_dict = {key: color for key, color in zip(keys, sns.color_palette("hls", len(keys)))}

n_hidden_layers_lst = sorted(np.unique(df.n_hidden_layers))

# lop off the first three numbers since they were tests and not fully run and trained
# df = df[3:]

# grossing formatting I have do to since I didn't take care to save the numbers in a float format
# df['posterior_predictive_density'] = (
#     df['posterior_predictive_density'].apply(lambda s: float(s.split('tensor(')[1].split(')')[0].split(',')[0])))

fig, axes = plt.subplots(3, 1, figsize=(4, 12), sharex=True, sharey=True)
for i, dataset in enumerate(['train', 'valid', 'test']):
    df_data = df[df.dataset == dataset]
    ax = axes[i]
    ax.set_xlabel('Number of hidden units per layer')
    ax.set_ylabel('Posterior Predictive Log Likelihood')
    if dataset == 'train':
        ax.set_title('Training Data')
    elif dataset == 'valid':
        ax.set_title('In-sample Holdout Data')
    elif dataset == 'test':
        ax.set_title('Out-of-sample Holdout Data')
    for inference in ['vae', 'svi', 'mcmc']:
        df_data_inference = df_data[df_data.inference == inference]
        if inference in ['svi', 'mcmc']:
            # mean_results = df_data_inference['posterior_predictive_density'].mean()
            # sd_results = df_data_inference['posterior_predictive_density'].std()
            key = inference.upper()
            # print(inference)
            # print(mean_results)
            # print(sd_results)
            max_results = df_data_inference['posterior_predictive_density'].max()
            ax.axhline(max_results, xmin=.05, xmax=.95, label=inference, color=colors_dict[key], linestyle='--')
            # ax.axhline(mean_results, xmin=.05, xmax=.95, label=inference, color=colors_dict[key], linestyle='--')
            # ax.fill_between(range(10, 100), mean_results - 2 * sd_results, mean_results + 2 * sd_results, color=colors_dict[key], alpha=0.4)
        else:
            for n_hidden_layers in (n_hidden_layers_lst):
                line = df_data_inference[df_data_inference.n_hidden_layers == n_hidden_layers]
                mean_results = line.groupby('n_hidden_units')['posterior_predictive_density'].apply(np.mean)
                sd_results = line.groupby('n_hidden_units')['posterior_predictive_density'].apply(np.std)
                key = inference.upper() + ' ({} hidden layer{})'.format(n_hidden_layers, 's' if n_hidden_layers > 1 else '')
                ax.errorbar(mean_results.index, mean_results.values, yerr=sd_results, color=colors_dict[key],
                            label=key)
    # each Axes object will have the same handles and labels
handles, labels = axes[0].get_legend_handles_labels()
# the hard-coded numbers scale with the size of the plot
legend = axes[-1].legend(handles, labels, loc='best', bbox_transform=fig.transFigure)
fig.tight_layout()
filename = os.path.join(results_dir, 'problem.pdf')
plt.savefig(filename, bbox_extra_artists=(legend,), bbox_inches='tight')

