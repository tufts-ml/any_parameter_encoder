import seaborn as sns
import pandas as pd


df = pd.read_csv('mdreviews1/reconstruction_by_sparsity_scale.csv')
g = sns.lineplot('num_topics', 'posterior_predictive_density', 'dataset', data=df)
fig = g.get_figure()
fig.savefig('mdreviews1/reconstruction_by_sparsity_scale.pdf')
