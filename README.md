# Any Parameter Encoder (APE)
Code to train and evaluate any-parameter encoder (APE).

To train and evaluate and APE on the toy bars dataset as mentioned in the paper, run
```python run.py --results_dir [results_dir] --architecture template --run_avi```

To additionally run the SVI and MCMC benchmarks, simply add the corresponding flags.

NOTE: This is still a work in progress. Not all benchmarks have been run on this code yet, but this version is meant to be a cleaned-up version to enable more effective future work.