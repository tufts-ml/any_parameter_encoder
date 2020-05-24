import numpy as np
import pandas as pd

df = pd.read_csv('no_training_non_toy_bars.csv')
architectures = ['template', 'template_unnorm', 'template_scaled', 'pseudo_inverse', 'pseudo_inverse_unnorm', 'pseudo_inverse_scaled']
df['architecture'] = np.tile(np.repeat(architectures, 4), int(len(df) / (len(architectures) * 4)))
print(df[df.model_type == 'avitm'].groupby(['metric', 'data_size', 'architecture']).loss.agg(mean=np.mean, std=np.std))
print(df.head(10))