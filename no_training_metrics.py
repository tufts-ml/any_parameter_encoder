import numpy as np
import argparse
import pandas as pd


parser = argparse.ArgumentParser(description='Results summary')
parser.add_argument('--csv', type=str)
args = parser.parse_args()
df = pd.read_csv(args.csv)
print(df.columns)
if 'architecture' not in list(df.columns):
    architectures = ['template', 'template_unnorm', 'template_scaled', 'pseudo_inverse', 'pseudo_inverse_unnorm', 'pseudo_inverse_scaled']
    df['architecture'] = np.tile(np.repeat(architectures, 4), int(len(df) / (len(architectures) * 4)))
print(df[df.model_type == 'avitm'].groupby(['metric', 'data_size', 'architecture']).loss.agg(mean=np.mean, std=np.std))