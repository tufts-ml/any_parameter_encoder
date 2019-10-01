import sys
import os
import math
import itertools
import numpy as np
import tensorflow as tf
import logging

from datasets.create import draw_random_doc
from utils import inverse_softmax, softmax, unzip_X_and_topics
from visualization.reconstructions import plot_side_by_side_docs


logger = logging.getLogger()

def create_minibatch(data, batch_size, shuffle=True):
    rng = np.random.RandomState(10)
    np.random.shuffle(data)
    logger.info('Finished shuffling training data.')
    for start_idx in range(0, len(data), batch_size):
        yield data[start_idx: start_idx + batch_size]

# def train(
#     train_data,
#     valid_data,
#     vae,
#     batch_size=200,
#     training_epochs=100,
#     display_step=5,
#     tensorboard=False,
#     tensorboard_logs_dir=None,
#     results_dir=None,
#     vae_meta=True,
#     shuffle=True,
#     vocab_size=100,
#     num_topics=20,
# ):
#     if tensorboard:
#         train_writer = tf.summary.FileWriter(
#             os.path.join(tensorboard_logs_dir, 'train'), vae.sess.graph)
#         valid_writer = tf.summary.FileWriter(
#             os.path.join(tensorboard_logs_dir, 'val'), vae.sess.graph)
#     train_docs, train_topics = train_data
#     valid_docs, valid_topics = valid_data
#     num_batches = int(math.ceil(len(train_docs) * len(train_topics) / batch_size))
#     for epoch in range(training_epochs):
#         costs_over_batches = []
#         for batch in range(num_batches):
#             vae.training_data = True
#             vae.sess.run(vae.train_iterator.initializer)
#             _, cost = vae.sess.run(
#                 (vae.optimizer, vae.cost),
#                 feed_dict={vae.keep_prob: 0.75},
#             )
#             costs_over_batches.append(cost)

#         if epoch % display_step == 0:
#             print(
#                 "Epoch: %04d" % (epoch + 1),
#                 "cost={:.9f}".format(sum(costs_over_batches)/len(costs_over_batches)),
#             )
#         if tensorboard:
#             merge = tf.summary.merge(vae.summaries)
#             if vae_meta:
#                 # summary = vae.sess.run(merge,
#                 #                         feed_dict={vae.x_placeholder: train_docs, vae.topics_placeholder: train_topics, vae.keep_prob: 1.0})
#                 summary = vae.sess.run(merge, feed_dict={vae.keep_prob: 1.0})
#             train_writer.add_summary(summary, epoch)
#             # vae.sess.run(vae.iterator.initializer, feed_dict={vae.x_placeholder: valid_docs, vae.topics_placeholder: valid_topics})
#             vae.training_data = False
#             vae.sess.run(vae.valid_iterator.initializer)
#             valid_cost = vae.sess.run(vae.cost, feed_dict={vae.keep_prob: 1.0})
#             valid_summary = tf.Summary(value=[tf.Summary.Value(tag="valid_loss", simple_value=valid_cost)])
#             valid_writer.add_summary(valid_summary, epoch)
#             valid_writer.flush()
#     return vae
    


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
    vae_meta=True,
    shuffle=True,
    save_iter=100,
    plot_valid_cost=True
):
    train_docs, train_topics = train_data
    train_data = list(itertools.product(train_docs, train_topics))
    logger.info('Train data created for training.')
    valid_docs, valid_topics = valid_data
    num_valid_topics = len(valid_topics)
    X_val = [d for d in valid_docs for _ in range(num_valid_topics)]
    topics_val = np.tile(valid_topics, (len(valid_docs), 1, 1))
    logger.info('Validation data created for training.')
    del valid_data
    logger.info('Deleted validation data.')
    if tensorboard:
        train_writer = tf.summary.FileWriter(
            os.path.join(tensorboard_logs_dir, 'train'), vae.sess.graph)
        valid_writer = tf.summary.FileWriter(
            os.path.join(tensorboard_logs_dir, 'val'), vae.sess.graph)

    # Training cycle
    for epoch in range(training_epochs):
        total_cost = 0.0
        num_batches = 0
        logger.info('Creating minibatches')
        for batch_xs in create_minibatch(train_data, batch_size, shuffle=shuffle):
            # Fit training using batch data
            cost = vae.partial_fit(batch_xs)
            logger.info('cost: {}'.format(cost))
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

        if epoch % save_iter == 0:
            vae.save()
        # Display logs per epoch step
        if epoch % display_step == 0:
            print(
                "Epoch: %04d" % (epoch + 1),
                "cost={:.9f}".format(total_cost / num_batches),
            )
            logger.info('Epoch: {}, cost: {}'.format(epoch + 1, total_cost / num_batches))
            if tensorboard:
                merge = tf.summary.merge(vae.summaries)
                if vae_meta:
                    summary = vae.sess.run(merge,
                                            feed_dict={vae.x: X, vae.topics: topics, vae.keep_prob: 1.0})
                else:
                    summary = vae.sess.run(merge,
                                            feed_dict={vae.x: X, vae.keep_prob: 1.0})
                train_writer.add_summary(summary, epoch)
                if plot_valid_cost:
                    if vae_meta:
                        valid_cost = vae.sess.run(vae.cost,
                                                feed_dict={vae.x: X_val, vae.topics: topics_val, vae.keep_prob: 1.0})
                    else:
                        valid_cost = vae.sess.run(vae.cost,
                                                feed_dict={vae.x: X_val, vae.keep_prob: 1.0})
                    print(valid_cost)
                    print('writing valid summary')
                    logger.info('writing valid summary: {}'.format(valid_cost))
                    valid_summary = tf.Summary(value=[tf.Summary.Value(tag="valid_loss", simple_value=valid_cost)])
                    valid_writer.add_summary(valid_summary, epoch)
                    valid_writer.flush()
    return vae


# def generate_data(vae, zs, vocab_size, num_docs=10000, min_n_words_per_doc=45, max_n_words_per_doc=60, d=0):
#     prng = np.random.RandomState(d)
#     reconstructed = vae.generate(inverse_softmax(zs))
#     # bring it to count space
#     new_docs = []
#     for i in range(num_docs):
#         new_doc = draw_random_doc(
#             reconstructed,
#             do_return_square=False, d=i
#         )
#         new_docs.append(new_doc)

#     new_docs = np.array(
#         [
#             np.bincount(doc.astype("int"), minlength=vocab_size)
#             for doc in new_docs
#             if np.sum(doc) != 0
#         ]
#     )
#     return np.array(new_docs)


def train_with_hallucinations(data, vae, model_config, alpha=.01, num_samples=10000,
                              batch_size=200, training_epochs=100, display_step=5,
                              tensorboard=False, tensorboard_logs_dir=None, results_dir=None):
    zs = np.random.dirichlet(alpha=alpha * np.ones(model_config['n_topics']), size=num_samples)
    fake_data = generate_data(vae, zs, vocab_size=model_config['vocab_size'], num_docs=100000)

    # plot_side_by_side_docs(fake_data, os.path.join(model_config['results_dir'], "fake_data.png"))

    vae = train(np.concatenate([data, fake_data]), vae, training_epochs=100, tensorboard=tensorboard, batch_size=batch_size,
                tensorboard_logs_dir=tensorboard_logs_dir, results_dir=results_dir)
    return vae


def find_lr(vae, data, batch_size, init_value=1e-8, final_value=10., beta=0.98):
    num = float(len(data)) / batch_size - 1
    print(len(data))
    print(batch_size)
    print(num)
    mult = (float(final_value) / init_value) ** (1/num)
    print('mult', mult)
    lr = init_value
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    for batch in create_minibatch(data, batch_size):
        batch_num += 1
        #As before, get the loss for this mini-batch of inputs/outputs
        loss = vae.evaluate(batch)
        print(loss)
        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) * loss
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        #Do the SGD step
        vae.partial_fit(batch, learning_rate=lr)
        #Update the lr for the next step
        lr *= mult
        print('new lr', lr)
    return log_lrs, losses