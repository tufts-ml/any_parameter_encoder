import argparse
import os
import numpy as np
import scipy.sparse
from distutils.dir_util import mkpath
from sklearn.externals import joblib
from bnpy.viz.BarsViz import show_square_images
import matplotlib.pyplot as plt

vocab_list = np.asarray([
    ['needle', 'finance',    'tech'],
    ['river',     'bank',  'stream'],
    ['mineral',   'gold', 'silicon'],
    ]).flatten().tolist()

tA = np.asarray([
    [.00, .00, .00],
    [.16, .16, .16],
    [.16, .16, .16],
    ])
tB = np.asarray([
    [.00, .16, .16],
    [.00, .16, .16],
    [.00, .16, .16],
    ])
tC = np.asarray([
    [.00, .00, .00],
    [.33, .33, .33],
    [.00, .00, .00],
    ])
tD = np.asarray([
    [.00, .00, .00],
    [.00, .00, .00],
    [.33, .33, .33],
    ])
tE = np.asarray([
    [.00, .33, .00],
    [.00, .33, .00],
    [.00, .33, .00],
    ])
tF = np.asarray([
    [.00, .00, .33],
    [.00, .00, .33],
    [.00, .00, .33],
    ])
tG = np.asarray([
    [.00, .33, .00],
    [.33, .33, .33],
    [.00, .33, .00],
    ])
tH = np.asarray([
    [.00, .00, .33],
    [.00, .00, .33],
    [.33, .33, .33],
    ])
# proba_list = [.38, .38, .08, .08, .02, .02, .02, .02]
topic_list = [tA, tB, tC, tD, tE, tF, tG, tH]
num_topics = len(topic_list)
proba_list = [1.0/num_topics for _ in topic_list]
for t in topic_list:
    t /= t.sum()
# tI is a topic only existing in the test set
# but is made up of constituent topics from up above
# instead of a never-before seen topic
tI = np.asarray([
    [.00, .00, .33],
    [.33, .33, .33],
    [.00, .00, .33],
    ])
tJ = np.asarray([
    [.33, .00, .00],
    [.33, .00, .00],
    [.33, .00, .00],
    ])
tK = np.asarray([
    [.33, .00, .00],
    [.33, .33, .33],
    [.33, .00, .00],
    ])
tL = np.asarray([
    [.00, .33, .00],
    [.00, .33, .00],
    [.33, .33, .33],
    ])
new_topic_list = [tI, tJ, tK, tL]
num_new_topics = len(new_topic_list)
new_proba_list = [1.0/num_new_topics for _ in new_topic_list]
for t in new_topic_list:
    t /= t.sum()

# save a plot of the bars
show_square_images(np.array(topic_list).reshape((num_topics, 9)), vmin=0, vmax=1)
plt.tight_layout()
plt.savefig('original_topics.png')

def draw_random_doc(
        topic_list,
        proba_list,
        min_n_words_per_doc=45,
        max_n_words_per_doc=60,
        do_return_square=True,
        proba_positive_label=0.2,
        d=0):
    prng = np.random.RandomState(d)
    V = topic_list[0].size

    # Pick which template
    # Each document is only in one topic
    k = prng.choice(len(proba_list), p=proba_list)
    n_words = prng.randint(low=min_n_words_per_doc, high=max_n_words_per_doc)
    words = prng.choice(
        V,
        p=topic_list[k].flatten(),
        replace=True,
        size=n_words)
    return words

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default=os.path.abspath('.'), type=str)
    parser.add_argument("--n_docs_train", default=100000, type=int)
    parser.add_argument("--n_docs_test", default=5000, type=int)
    parser.add_argument("--n_docs_valid", default=5000, type=int)

    args = parser.parse_args()
    dataset_path = os.path.abspath(args.dataset_path)

    x_list = list()
    n_docs = args.n_docs_train + args.n_docs_valid
    for d in range(n_docs):
        x_V = draw_random_doc(
            topic_list,
            proba_list,
            do_return_square=False,
            d=d,
            )
        x_list.append(x_V)
        if (d+1) % 1000 == 0 or (d == n_docs -1):
            print "generated doc %d/%d" % (d+1, n_docs)

    # add the new topic docs to the end so they are part of the test set only
    for d in range(args.n_docs_test):
        x_V = draw_random_doc(
            new_topic_list,
            new_proba_list,
            do_return_square=False,
            d=d,
            )
        x_list.append(x_V)
        if (d+1) % 1000 == 0 or (d == n_docs -1):
            print "generated new dist doc %d/%d" % (d+1, n_docs)

    x_list = np.array(x_list)
    train_doc_ids = np.arange(args.n_docs_train)
    valid_doc_ids = np.arange(
        args.n_docs_train,
        args.n_docs_train + args.n_docs_valid)
    test_doc_ids = np.arange(
        args.n_docs_train + args.n_docs_valid,
        x_list.shape[0])
    
    # make sure that only topic I is in the test set
    # make sure that only the test set contains topic I
    tI_words = [0, 3, 6]
    non_tI_words = [word for word in range(topic_list[0].size)
                    if word not in tI_words]
    assert all([any(np.isin(non_tI_words, doc))
                for doc in x_list[train_doc_ids]])
    assert all([any(np.isin(non_tI_words, doc))
                for doc in x_list[valid_doc_ids]])
    assert all([any(np.isin(non_tI_words, doc, invert=True))
                for doc in x_list[test_doc_ids]])

    np.save(os.path.join(dataset_path, "train.txt.npy"), x_list[train_doc_ids])
    np.save(os.path.join(dataset_path, "valid.txt.npy"), x_list[valid_doc_ids])
    np.save(os.path.join(dataset_path, "test.txt.npy"), x_list[test_doc_ids])
