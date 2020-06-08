# Any Parameter Encoder (APE)
Code to train and evaluate any-parameter encoder (APE).

To train and evaluate and APE on the toy bars dataset as mentioned in the paper, run
```python run.py --results_dir [results_dir] --architecture template --run_avi```

To additionally run the SVI and MCMC benchmarks, simply add the corresponding flags.

## Training from scratch

To train APE_VAE (e.g. encoder architecture is aware of the decoder params) or the standard VAE from scratch, run
```python run_ape.py --results_dir [results_dir] --architecture template```