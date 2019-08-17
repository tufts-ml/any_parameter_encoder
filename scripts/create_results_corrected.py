import os
import pandas as pd

# this corrects for a problem in the vae results on the training dataset
# and takes the max of 5 SVI runs on each dataset


def take_max_run(df, columns):
    df_means = df.groupby(columns).posterior_predictive_density.mean()
    levels = range(len(columns) - 1)
    df_best_params = df_means.groupby(level=levels).idxmax().values.flatten()
    rows = []
    for params in df_best_params:
        subset = df.copy()
        for col, param in zip(columns, params):
            subset = subset[subset[col] == param]
        rows.append(subset)
    df = pd.concat(rows)
    return df

# we take the MCMC results from the initial full run
df = pd.read_csv('problem_toy_bars/results.csv', header=None)
# we take the VAE results from a separate run that corrected for weird results in training
# df1 = pd.read_csv('problem_toy_bars/results_train.csv', header=None)
# we take the SVI results from a run where we take the max of 5 runs
df2 = pd.read_csv('svi_baseline/results.csv', header=None)
# we take the VAE results from a run where we take the max of 5 runs
vae_dfs = []
for i in range(1, 5):
    run_results = pd.read_csv(os.path.join('vae_run' + str(i), 'results.csv'), header=None)
    run_results['run'] = i
    vae_dfs.append(run_results)
df3 = pd.concat(vae_dfs)

df.columns = ['inference', 'model', 'dataset', 'n_hidden_layers', 'n_hidden_units', 'posterior_predictive_density']
# df1.columns = ['inference', 'model', 'dataset', 'n_hidden_layers', 'n_hidden_units', 'posterior_predictive_density']
df2.columns = ['inference', 'model', 'dataset', 'n_hidden_layers', 'n_hidden_units', 'posterior_predictive_density']
df3.columns = ['inference', 'model', 'dataset', 'n_hidden_layers', 'n_hidden_units', 'posterior_predictive_density', 'run']

df = df[(df.inference == 'mcmc')]

# df1 = df1[df1.model.isin(['lda_orig', 'lda_scale', 'lda_orig_hallucinations', 'lda_scale_hallucinations'])]

# here, n_hidden_layers functions as the "run" index (a hack)
df2 = take_max_run(df2, ['dataset', 'n_hidden_layers'])

df3 = take_max_run(df3, ['dataset', 'model','n_hidden_layers', 'n_hidden_units', 'run'])
# df3 = df3[df3.run==2]
df3.drop(['run'], axis=1, inplace=True)

df_corrected = pd.concat([df, df2, df3])
df_corrected.to_csv('problem_toy_bars/results_corrected_full.csv', header=None, index=False)