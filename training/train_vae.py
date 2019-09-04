import sys
import os
import math
import numpy as np
import tensorflow as tf

from datasets.create import draw_random_doc
from utils import inverse_softmax, softmax, unzip_X_and_topics
from visualization.reconstructions import plot_side_by_side_docs


def create_minibatch(data, batch_size):
    rng = np.random.RandomState(10)
    np.random.shuffle(data)
    for start_idx in range(0, len(data), batch_size):
        yield data[start_idx: start_idx + batch_size]


def train(
    train_data,
    valid_data,
    vae,
    batch_size=200,
    training_epochs=100,
    display_step=5,
    tensorboard=False,
    tensorboard_logs_dir=None,
    results_dir=None,
    vae_meta=True
):
    if tensorboard:
        train_writer = tf.summary.FileWriter(
            os.path.join(tensorboard_logs_dir, 'train'), vae.sess.graph)
        valid_writer = tf.summary.FileWriter(
            os.path.join(tensorboard_logs_dir, 'val'), vae.sess.graph)

    # Training cycle
    for epoch in range(training_epochs):
        total_cost = 0.0
        num_batches = 0
        # Loop over all batches
        for batch_xs in create_minibatch(train_data, batch_size):
            # Fit training using batch data
            cost = vae.partial_fit(batch_xs)
            # Keep track of the number of batches
            num_batches += 1
            # Keep track of the loss
            total_cost += cost
            X, topics = unzip_X_and_topics(batch_xs)

            if np.isnan(total_cost):
                if vae_meta:
                    z = vae.sess.run(vae.z,
                                     feed_dict={vae.x: X, vae.topics: topics, vae.keep_prob: 1.0})
                else:
                    z = vae.sess.run(vae.z,
                                   feed_dict={vae.x: X, vae.keep_prob: 1.0})
                print(z)
                print(z.shape)      
                print(epoch, np.sum(X, 1).astype(np.int), X.shape)
                print(cost)
                print(
                    "Encountered NaN, stopping training. "
                    "Please check the learning_rate settings and the momentum."
                )
                # return vae,emb
                sys.exit()
        if not vae_meta and epoch < 15:
            topics = softmax(vae.topic_prop(batch_xs))
            plot_side_by_side_docs(topics, os.path.join(results_dir, 'topics_{}.pdf'.format(str(epoch).zfill(2))))
            recreated_docs, _, _ = vae.recreate_input(batch_xs[:10])
            X, topics = unzip_X_and_topics(batch_xs[:10])
            plot_side_by_side_docs(np.concatenate([X, recreated_docs]), os.path.join(results_dir, 'recreated_docs_{}.pdf'.format(str(epoch).zfill(2))))

        # Display logs per epoch step
        if epoch % display_step == 0:
            print(
                "Epoch: %04d" % (epoch + 1),
                "cost={:.9f}".format(total_cost / num_batches),
            )
            if tensorboard:
                merge = tf.summary.merge(vae.summaries)
                if vae_meta:
                    summary = vae.sess.run(merge,
                                            feed_dict={vae.x: X, vae.topics: topics, vae.keep_prob: 1.0})
                else:
                    summary = vae.sess.run(merge,
                                            feed_dict={vae.x: X, vae.keep_prob: 1.0})
                train_writer.add_summary(summary, epoch)
                X_val, topics_val = unzip_X_and_topics(valid_data)
                if vae_meta:
                    valid_cost = vae.sess.run(vae.cost,
                                            feed_dict={vae.x: X_val, vae.topics: topics_val, vae.keep_prob: 1.0})
                else:
                    valid_cost = vae.sess.run(vae.cost,
                                            feed_dict={vae.x: X_val, vae.keep_prob: 1.0})
                print(valid_cost)
                print('writing valid summary')
                valid_summary = tf.Summary(value=[tf.Summary.Value(tag="valid_loss", simple_value=valid_cost)])
                valid_writer.add_summary(valid_summary, epoch)
                valid_writer.flush()
    return vae


def generate_data(vae, zs, vocab_size, num_docs=10000, min_n_words_per_doc=45, max_n_words_per_doc=60, d=0):
    prng = np.random.RandomState(d)
    reconstructed = vae.generate(inverse_softmax(zs))
    # bring it to count space
    new_docs = []
    for i in range(num_docs):
        new_doc = draw_random_doc(
            reconstructed,
            do_return_square=False, d=i
        )
        new_docs.append(new_doc)

    new_docs = np.array(
        [
            np.bincount(doc.astype("int"), minlength=vocab_size)
            for doc in new_docs
            if np.sum(doc) != 0
        ]
    )
    return np.array(new_docs)


def train_with_hallucinations(data, vae, model_config, alpha=.01, num_samples=10000,
                              batch_size=200, training_epochs=100, display_step=5,
                              tensorboard=False, tensorboard_logs_dir=None, results_dir=None):
    zs = np.random.dirichlet(alpha=alpha * np.ones(model_config['n_topics']), size=num_samples)
    fake_data = generate_data(vae, zs, vocab_size=model_config['vocab_size'], num_docs=100000)

    # plot_side_by_side_docs(fake_data, os.path.join(model_config['results_dir'], "fake_data.png"))

    vae = train(np.concatenate([data, fake_data]), vae, training_epochs=100, tensorboard=tensorboard, batch_size=batch_size,
                tensorboard_logs_dir=tensorboard_logs_dir, results_dir=results_dir)
    return vae