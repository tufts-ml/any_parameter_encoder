import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


results_dir = 'dump_fixed2'
results_csv = 'results.csv'

df = pd.read_csv(os.path.join(results_dir, results_csv), header=None)
df.columns = ['inference', 'model', 'dataset', 'n_hidden_layers', 'n_hidden_units', 'posterior_predictive_density']

keys = ['SVI', 'MCMC', 'VAE (1 hidden layer)', 'VAE (2 hidden layers)', 'VAE (5 hidden layers)']
colors_dict = {key: color for key, color in zip(keys, sns.color_palette("hls", len(keys)))}

# lop off the first three numbers since they were tests and not fully run and trained
# df = df[3:]

# grossing formatting I have do to since I didn't take care to save the numbers in a float format
# df['posterior_predictive_density'] = (
#     df['posterior_predictive_density'].apply(lambda s: float(s.split('tensor(')[1].split(')')[0].split(',')[0])))

fig, axes = plt.subplots(3, 1)
for i, dataset in enumerate(['train', 'valid', 'test']):
    df_data = df[df.dataset == dataset]
    ax = axes[i]
    for inference in ['vae', 'svi', 'mcmc']:
        df_data_inference = df_data[df_data.inference == inference]
        if inference in ['svi', 'mcmc']:
            mean_results = df_data_inference['posterior_predictive_density'].mean()
            sd_results = df_data_inference['posterior_predictive_density'].std()
            key = inference.upper()
            ax.axhline(mean_results, label=inference, color=colors_dict[key])
            ax.fill_between(range(101), mean_results - 2 * sd_results, mean_results + 2 * sd_results, color=colors_dict[key], alpha=0.4)
        else:
            for n_hidden_layers in ([1, 2, 5]):
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

