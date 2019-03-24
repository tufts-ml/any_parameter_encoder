import os
import numpy as np


def load_toy_bars(datadir, VOCAB_SIZE=9):
    ### Step 1: Load data
    dataset_tr = os.path.join("datasets", datadir, "train.txt.npy")
    data_tr = np.load(dataset_tr)

    dataset_va = os.path.join("datasets", datadir, "valid.txt.npy")
    data_va = np.load(dataset_va)

    dataset_te = os.path.join("datasets", datadir, "test.txt.npy")
    data_te = np.load(dataset_te)

    ### Step 2: Convert data to counts
    data_tr = np.array(
        [
            np.bincount(doc.astype("int"), minlength=VOCAB_SIZE)
            for doc in data_tr
            if np.sum(doc) != 0
        ]
    )
    data_va = np.array(
        [
            np.bincount(doc.astype("int"), minlength=VOCAB_SIZE)
            for doc in data_va
            if np.sum(doc) != 0
        ]
    )
    data_te = np.array(
        [
            np.bincount(doc.astype("int"), minlength=VOCAB_SIZE)
            for doc in data_te
            if np.sum(doc) != 0
        ]
    )
    return data_tr, data_va, data_te