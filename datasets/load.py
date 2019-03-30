import os
import numpy as np


def load_toy_bars(datadir, VOCAB_SIZE=100):
    ### Step 1: Load data
    data_files = ["train.txt.npy", "valid.txt.npy", "test.txt.npy", "test_single.txt.npy", "test_double.txt.npy", "test_triple.txt.npy"]
    datasets = []
    for data_file in data_files:
        data = np.load(os.path.join("datasets", datadir, data_file))
        dataset = np.array(
            [
                np.bincount(doc.astype("int"), minlength=VOCAB_SIZE)
                for doc in data
                if np.sum(doc) != 0
            ]
        )
        datasets.append(dataset)
    return datasets