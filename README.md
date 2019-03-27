The code is organized in the following way:

1. VAE training: takes in parameters, outputs h5 files of weights
2. Evaluation of inference methods: takes in weights, outputs metrics to a results csv file and saves examples as pdfs
3. Visualization: takes in csv results file, creates the plot

To complete 1 and 2, run one of the main "run" files. I suggest `run_fixed.py`, which initializes and fixes the VAE encoder. You can specify your directory and file names, as well as the parameters, at the top of the page. Example:

`python run_fixed.py`

Since the run can be long, a command like nohup is very useful here. Example:

`nohup python -u run_fixed.py > run_fixed.out &`   

To complete 3, run `python visualize/full_results.py`. This has not been fully generalized yet, so you might have to manually update some of the values that are currently hardcoded in, but code for this should be completed shortly, whereby you can just specify your csv file of results and then create the plots.   