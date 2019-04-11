import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math


results_dir = 'problem_toy_bars'
# results_dir = 'dump_fixed2'
results_csv = 'results_corrected_full.csv'
plot_name = 'problem_corrected_full.pdf'
# results_csv = 'results.csv'
# plot_name = 'problem_full.pdf'


df = pd.read_csv(os.path.join(results_dir, results_csv), header=None)
df.columns = ['inference', 'model', 'dataset', 'n_hidden_layers', 'n_hidden_units', 'posterior_predictive_density']
df = df[df.model.isin(['lda_orig', 'lda_scale', 'lda_orig_hallucinations', 'lda_scale_hallucinations'])]

datasets = ['train', 'valid', 'test', 'test_single', 'test_double', 'test_triple']
inferences = np.unique(df.inference)
models = np.unique(df.model)
n_hidden_layers_lst = sorted(np.unique(df[df.inference=='vae'].n_hidden_layers))

keys = {'svi': 'SVI', 'mcmc': 'MCMC', 'mcmc_lda': 'MCMC (LDA)'}
for model in models:
    for n_hidden_layers in n_hidden_layers_lst:
        keys.update(
            {(model, n_hidden_layers):
                 'VAE ' + model.upper().replace('_', ' ') + ' ({} hidden layers)'.format(
                     n_hidden_layers, 's' if n_hidden_layers > 1 else '')})

colors_dict = {n_layers: color for n_layers, color in zip(n_hidden_layers_lst, sns.color_palette("hls", len(n_hidden_layers_lst)))}
colors_dict.update({
    'svi': 'silver',
    'mcmc': 'gold'
})
linestyles_dict = {
    'lda_orig': '-',
    'lda_scale': (0, (5, 10)),  # loosely dashed
    'lda_orig_hallucinations': (0, (1, 10)),  # loosely dotted
    'lda_scale_hallucinations': (0, (3, 5, 1, 5))  # dash dotted
}

# lop off the first three numbers since they were tests and not fully run and trained
# df = df[3:]

# grossing formatting I have do to since I didn't take care to save the numbers in a float format
# df['posterior_predictive_density'] = (
#     df['posterior_predictive_density'].apply(lambda s: float(s.split('tensor(')[1].split(')')[0].split(',')[0])))
n_rows = 3
n_cols = int(math.ceil(len(datasets)/3))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_rows * 6, n_cols * 8), sharex=True, sharey=True)
if n_cols == 1:
    axes = np.expand_dims(axes, 1)
for i, dataset in enumerate(datasets):
    df_data = df[df.dataset == dataset]
    ax = axes[i%3][i/3]
    ax.set_xlabel('Number of hidden units per layer')
    ax.set_ylabel('Posterior Predictive Log Likelihood')
    ax.set_ylim([-150000, -70000])
    if dataset == 'train':
        ax.set_title('Training Data')
    elif dataset == 'valid':
        ax.set_title('In-sample Holdout Data')
    elif dataset == 'test':
        ax.set_title('Out-of-sample Holdout Data')
    elif dataset == 'test_single':
        ax.set_title('Out-of-sample Holdout Data (Single bars only)')
    elif dataset == 'test_double':
        ax.set_title('Out-of-sample Holdout Data (Double bars only)')
    elif dataset == 'test_triple':
        ax.set_title('Out-of-sample Holdout Data (Triple bars only)')
    for inference in inferences:
        df_data_inference = df_data[df_data.inference == inference]
        if inference in ['svi', 'mcmc', 'mcmc_lda']:
            # if multiple runs, choose the run with the best mean
            grouped = df_data_inference.groupby(['n_hidden_layers', 'n_hidden_units'])
            results = grouped['posterior_predictive_density'].agg({'mean':'mean', 'std':'std'})
            try:
                max_idx = results['mean'].idxmax()
                mean_results = results.loc[max_idx]['mean']
                sd_results = results.loc[max_idx]['std']
                ax.axhline(mean_results, xmin=.05, xmax=.95, label=inference, color=colors_dict[inference], linestyle='--')
                ax.fill_between(range(10, 100), mean_results - 2 * sd_results, mean_results + 2 * sd_results,
                                color=colors_dict[inference], alpha=0.4)
            except:
                print(dataset, inference)
                print(results.shape)
        else:
            for model in models:
                df_data_inference_model = df_data_inference[df_data_inference.model==model]
                for n_hidden_layers in (n_hidden_layers_lst):
                    line = df_data_inference_model[df_data_inference_model.n_hidden_layers == n_hidden_layers]
                    mean_results = line.groupby('n_hidden_units')['posterior_predictive_density'].apply(np.mean)
                    sd_results = line.groupby('n_hidden_units')['posterior_predictive_density'].apply(np.std)
                    ax.errorbar(mean_results.index, mean_results.values, yerr=sd_results,
                                color=colors_dict[n_hidden_layers], label=keys[model, n_hidden_layers],
                                linestyle=linestyles_dict.get(model, '-'))
# each Axes object will have the same handles and labels
handles, labels = axes[0][0].get_legend_handles_labels()
# the hard-coded numbers scale with the size of the plot
legend = axes[-1][-1].legend(handles, labels, loc='best', bbox_transform=fig.transFigure, handlelength=3)
fig.tight_layout()
filename = os.path.join(results_dir, plot_name)
plt.savefig(filename, bbox_extra_artists=(legend,), bbox_inches='tight')

