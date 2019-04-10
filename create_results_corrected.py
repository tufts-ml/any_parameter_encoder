import pandas as pd

# this corrects for a problem in the vae results on the training dataset
# and takes the max of 5 SVI runs on each dataset

# we take the MCMC results from the initial full run
df = pd.read_csv('problem_toy_bars/results.csv')
# we take the VAE results from a separate run that corrected for weird results in training
df1 = pd.read_csv('problem_toy_bars/results_train.csv')
# we take the SVI results from a run where we take the max of 5 runs
df2 = pd.read_csv('svi_baseline/results.csv')
df.columns = ['inference', 'model', 'dataset', 'n_hidden_layers', 'n_hidden_units', 'posterior_predictive_density']
df1.columns = ['inference', 'model', 'dataset', 'n_hidden_layers', 'n_hidden_units', 'posterior_predictive_density']
df2.columns = ['inference', 'model', 'dataset', 'n_hidden_layers', 'n_hidden_units', 'posterior_predictive_density']
df = df[(df.inference == 'mcmc')]
df1 = df1[df1.model.isin(['lda_orig', 'lda_scale', 'lda_orig_hallucinations', 'lda_scale_hallucinations'])]
df2_means = df2.groupby(['dataset', 'n_hidden_layers']).posterior_predictive_density.mean()
df2_best_params = df2_means.groupby(level=0).idxmax().values.flatten()
rows = []
for params in df2_best_params:
    rows.append(df2[(df2.dataset==params[0])&(df2.n_hidden_layers==params[1])])
df2 = pd.concat(rows)
df_corrected = pd.concat([df, df1, df2])
df_corrected.to_csv('problem_toy_bars/results_corrected_full.csv', header=None, index=False)