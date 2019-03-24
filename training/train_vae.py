import sys
import math
import numpy as np
import tensorflow as tf


def create_minibatch(data, batch_size):
    rng = np.random.RandomState(10)
    shuffled_data = np.random.shuffle(data)
    for start_idx in range(0, data.shape[0], batch_size):
        yield data[start_idx: start_idx + batch_size]



def train(
    data,
    vae,
    batch_size=200,
    training_epochs=100,
    display_step=5,
    tensorboard=False,
    tensorboard_logs_dir=None
):
    if tensorboard:
        train_writer = tf.summary.FileWriter(
            tensorboard_logs_dir, vae.sess.graph)

    # Training cycle
    for epoch in range(training_epochs):
        total_cost = 0.0
        num_batches = 0
        # Loop over all batches
        for batch_xs in create_minibatch(data.astype("float32"), batch_size):
            # Fit training using batch data
            cost = vae.partial_fit(batch_xs)
            # Keep track of the number of batches
            num_batches += 1
            # Keep track of the loss
            total_cost += cost

            if np.isnan(total_cost):
                print(vae.sess.run(vae.z,
                                   feed_dict={vae.x: batch_xs, vae.keep_prob: 1.0}))
                print(epoch, np.sum(batch_xs, 1).astype(np.int), batch_xs.shape)
                print(cost)
                print(
                    "Encountered NaN, stopping training. "
                    "Please check the learning_rate settings and the momentum."
                )
                # return vae,emb
                sys.exit()

        # Display logs per epoch step
        if epoch % display_step == 0:
            # print(vae.sess.run(vae.layer_do_0,
            #     feed_dict={vae.x: batch_xs, vae.keep_prob: 1.0}))
            print(
                "Epoch: %04d" % (epoch + 1),
                "cost={:.9f}".format(total_cost / num_batches),
            )

        if tensorboard:
            merge = tf.summary.merge(vae.summaries)
            summary = vae.sess.run(merge,
                                   feed_dict={vae.x: batch_xs, vae.keep_prob: 1.0})
            train_writer.add_summary(summary, epoch)

    return vae