import os
import h5py
import numpy as np
import torch
from torch import autograd
import torch.nn as nn

import pyro
import pyro.distributions as dist

import numpy as np
import tensorflow as tf
from functools import partial
from utils import unzip_X_and_topics
import logging

logger = logging.getLogger()
l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
cwd = os.getcwd()

def linear_warmup_and_cooldown(starting_learning_rate, global_step, decay_steps, decay_rate):
    """
    Arguments:
        starting_learning_rate is the max learning rate
        decay_steps is the number of steps before we switch from warm up to cool down
        decay_rate is the rate we decay during cooldown
    """
    if global_step < decay_steps:
        return global_step / decay_steps * starting_learning_rate
    return starting_learning_rate * (1 - (global_step - decay_steps) * decay_rate)

def one_cycle_policy(lr_max, global_step, tot_epochs, num_batches, 
                     momentum_min=0.85, momentum_max=0.95, pct_start=0.3, div_factor=25,
                     final_div=1e4):
    # n = num_batches * tot_epochs
    # a1 = int(n * pct_start)
    # a2 = n - a1
    # lr_min = tf.cond(
    #     tf.math.less_equal(global_step, a1),
    #     lambda: lr_max / div_factor,
    #     lambda: lr_max / (div_factor * final_div)
    # )
    # momentum = tf.cond(
    #     tf.math.less_equal(global_step, a1),
    #     lambda: momentum_max - tf.math.multiply(momentum_max - momentum_min, tf.cast(tf.math.divide(global_step, a1), tf.float32)),
    #     lambda: momentum_min + tf.math.multiply(momentum_max - momentum_min, tf.cast(tf.math.divide(global_step,  a2), tf.float32))
    # )
    # lr = tf.cond(
    #     tf.math.less_equal(global_step, a1),
    #     lambda: lr_min + tf.math.multiply(lr_max - lr_min, tf.cast(tf.math.divide(global_step, a2), tf.float32)),
    #     lambda: lr_max - tf.math.multiply(lr_max - lr_min, tf.cast(tf.math.divide(global_step, a1), tf.float32)),
    # )
    # print('learning_rate', lr)
    # print('momentum', momentum)
    # return tf.cast(lr, tf.float32), tf.cast(momentum, tf.float32)
    n = num_batches * tot_epochs
    a1 = int(n * pct_start)
    a2 = n - a1
    if global_step < a1:
        lr_min = lr_max / div_factor
        learning_rate = lr_min + (lr_max - lr_min) * global_step / a1
        momentum = momentum_max - (momentum_max - momentum_min) * global_step / a1
    else:
        lr_min = lr_max / (div_factor * final_div)
        learning_rate = lr_max - (lr_max - lr_min) * global_step / a2
        momentum = momentum_min + (momentum_max - momentum_min) * global_step / a2
    print('learning_rate', learning_rate)
    print('momentum', momentum)
    return tf.cast(learning_rate, tf.float32), tf.cast(momentum, tf.float32)


class VAE_tf(object):
    """
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """

    def __init__(
            self,
            n_hidden_units=100,
            n_hidden_layers=2,
            n_topics=4,
            model_name=None,
            results_dir=None,
            vocab_size=9,
            scale_trainable=False,
            architecture="naive",
            transfer_fct=tf.nn.softplus,
            starting_learning_rate=0.002,
            decay_steps=1000,
            decay_rate=.9,
            n_samples=1,
            batch_size=200,
            alpha=1,
            n_steps_enc=1,
            tensorboard=False,
            custom_lr=False,
            use_dropout=False,
            use_adamw = False,
            scale_type='sample',
            test_lr=False,
            tot_epochs=None,
            num_batches=None,
            seed=0,
            skip_connections=False,
            **kwargs
    ):
        self.n_hidden_units = n_hidden_units
        self.n_hidden_layers = n_hidden_layers
        self.n_topics = n_topics
        self.model_name = model_name
        self.results_dir = results_dir
        self.vocab_size = vocab_size
        self.scale_trainable = scale_trainable
        self.architecture = architecture
        self.transfer_fct = transfer_fct
        self.starting_learning_rate = starting_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.n_steps_enc = n_steps_enc
        self.tensorboard = tensorboard
        self.custom_lr = custom_lr
        self.use_dropout = use_dropout
        self.use_adamw = use_adamw
        self.scale_type = scale_type
        # only used for one-cycle policy
        self.test_lr = test_lr
        self.momentum = None
        self.tot_epochs = tot_epochs
        self.num_batches = num_batches
        tf.random.set_random_seed(seed)
        self.skip_connections = skip_connections
        if self.custom_lr:
            self.global_step = 0
            # self.global_step = tf.Variable(0, trainable=False, name='global_step')
        else:
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.summaries = []
        """----------------Inputs----------------"""
        # def generate_batches(data_name):
        #     topics = np.load(os.path.join(results_dir, '{}_topics.npy'.format(data_name)))
        #     documents = np.load(os.path.join(results_dir, 'documents.npy'))
        #     np.random.shuffle(documents)
        #     np.random.shuffle(topics)
        #     for doc in documents:
        #         for t in topics:
        #             yield doc, t
        # self.training_data = True
        # generate_train_batches = partial(generate_batches, 'train')
        # generate_valid_batches = partial(generate_batches, 'valid')
        # train_dataset = tf.data.Dataset.from_generator(
        #         generate_train_batches, (tf.float32, tf.float32),
        #         (tf.TensorShape([vocab_size]), tf.TensorShape([self.n_topics, vocab_size]))
        #     ).repeat().batch(batch_size).shuffle(buffer_size=1000)
        # valid_dataset = tf.data.Dataset.from_generator(
        #         generate_valid_batches, (tf.float32, tf.float32),
        #         (tf.TensorShape([vocab_size]), tf.TensorShape([self.n_topics, vocab_size]))
        #     ).repeat().batch(batch_size)
        # self.train_iterator = train_dataset.make_initializable_iterator()
        # self.valid_iterator = valid_dataset.make_initializable_iterator()
        # if self.training_data:
        #     iterator = self.train_iterator
        # else:
        #     iterator = self.valid_iterator
        # self.x, self.topics = iterator.get_next()


        self.x = tf.placeholder(tf.float32, [None, self.vocab_size], name='x')
        self.topics = tf.placeholder(tf.float32, [None, self.n_topics, self.vocab_size], name='topics')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        if self.test_lr:
            self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """-------Constructing Laplace Approximation to Dirichlet Prior--------------"""
        self.h_dim = float(self.n_topics)
        self.a = alpha * np.ones((1, int(self.h_dim))).astype(np.float32)
        self.mu2 = tf.constant((np.log(self.a).T - np.mean(np.log(self.a), 1)).T)
        self.var2 = tf.constant(
            (
                    ((1.0 / self.a) * (1 - (2.0 / self.h_dim))).T
                    + (1.0 / (self.h_dim * self.h_dim)) * np.sum(1.0 / self.a, 1)
            ).T
        )

        # Create autoencoder network
        # self.sess = tf.InteractiveSession()
        self._create_network()
        self._create_loss_optimizer()
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _create_network(self):
        self.network_weights = self._initialize_weights()

        self.z_mean, self.z_log_sigma_sq = self._recognition_network(
            self.network_weights["weights_recog"], self.network_weights["biases_recog"],
        )
        eps = tf.random_normal((self.n_samples, self.n_topics), 0, 1, dtype=tf.float32)
        if self.scale_type == 'mean':
            self.z_mean = tf.multiply(self.scale, self.z_mean)
        z = tf.add(
            tf.expand_dims(self.z_mean, axis=1), tf.multiply(tf.sqrt(tf.exp(tf.expand_dims(self.z_log_sigma_sq, axis=1))), eps)
        )
        if self.scale_type == 'sample':
            z = tf.multiply(self.scale, z)
        self.z = tf.reduce_mean(z, axis=1)
        self.sigma = tf.exp(self.z_log_sigma_sq)
        # generator = partial(self._generator_network, self.network_weights['weights_gener'])
        # self.x_reconstr_mean = tf.reduce_mean(tf.map_fn(generator, self.z), axis=1)
        self.x_reconstr_means = self._generator_network(z)
        self.x_reconstr_mean = tf.reduce_mean(self.x_reconstr_means, axis=1)

    def _initialize_weights(self):
        all_weights = dict()
        with tf.variable_scope("recognition_network"):
            all_weights["weights_recog"] = {
                "out_mean": tf.get_variable("out_mean", [self.n_hidden_units, self.n_topics]),
                "out_log_sigma": tf.get_variable("out_log_sigma", [self.n_hidden_units, self.n_topics]),
            }

            all_weights["biases_recog"] = {
                "b1": tf.get_variable("b1", [self.n_hidden_units], initializer=tf.zeros_initializer()),
                "out_mean": tf.get_variable("out_mean_b", [self.n_topics], initializer=tf.zeros_initializer()),
                "out_log_sigma": tf.get_variable("out_log_sigma_b", [self.n_topics], initializer=tf.zeros_initializer()),
            }

            # add more layers
            if self.n_hidden_layers > 1:
                for i in range(2, self.n_hidden_layers + 1):
                    all_weights["weights_recog"]["h{}".format(i)] = tf.get_variable(
                        "h{}".format(i), [self.n_hidden_units, self.n_hidden_units])
                    all_weights["biases_recog"]["b{}".format(i)] = tf.get_variable(
                        "b{}".format(i), [self.n_hidden_units], initializer=tf.zeros_initializer())
            # first layer is larger
            if self.architecture == "naive":
                all_weights["weights_recog"].update({
                    "h1": tf.get_variable(
                        "h1", [(1 + self.n_topics) * self.vocab_size, self.n_hidden_units])})
            elif self.architecture == "template":
                all_weights["weights_recog"].update({
                    "h1": tf.get_variable(
                        "h1", [self.n_topics, self.n_hidden_units])})
            elif self.architecture == "template_plus_topics":
                all_weights["weights_recog"].update({
                    "h1": tf.get_variable(
                        "h1", [self.n_topics * (1 + self.vocab_size), self.n_hidden_units])})
            elif self.architecture == "standard":
                all_weights["weights_recog"].update(
                    {"h1": tf.get_variable("h1", [self.vocab_size, self.n_hidden_units])})
            elif self.architecture == "naive_separated":
                if self.n_hidden_layers > 1:
                    raise NotImplementedError('This architecture only supports one hidden layer right now')
                all_weights["weights_recog"].update({
                    "h1_mu": tf.get_variable(
                        "h1_mu", [(1 + self.n_topics) * self.vocab_size, self.n_hidden_units])})
                all_weights["weights_recog"].update({
                    "h1_sigma": tf.get_variable(
                        "h1_sigma", [(1 + self.n_topics) * self.vocab_size, self.n_hidden_units])})
                all_weights["biases_recog"].update({
                    "b1_mu": tf.get_variable("b1_mu", [self.n_hidden_units], initializer=tf.zeros_initializer())})
                all_weights["biases_recog"].update({
                    "b1_sigma": tf.get_variable("b1_sigma", [self.n_hidden_units], initializer=tf.zeros_initializer())})
            else:
                raise ValueError("architecture must be either 'naive' or 'template'")

            self.scale = tf.Variable(tf.ones([]), name="scale", trainable=self.scale_trainable)

        return all_weights

    def _recognition_network(self, weights, biases):
        with tf.variable_scope("recognition_network"):
            if self.architecture == "naive":
                x_and_topics = tf.reshape(tf.concat([
                    tf.expand_dims(self.x, 1), self.topics
                    ], axis=1), (-1, (1 + self.n_topics) * self.vocab_size))
                layer = tf.contrib.layers.batch_norm(self.transfer_fct(
                    tf.add(tf.matmul(x_and_topics, weights["h1"]), biases["b1"])))
            elif self.architecture == "template":
                layer = tf.einsum("ab,abc->ac", self.x, tf.transpose(self.topics, perm=[0, 2, 1]))
                layer = tf.contrib.layers.batch_norm(self.transfer_fct(
                    tf.add(tf.matmul(layer, weights["h1"]), biases["b1"])))
            elif self.architecture == "template_plus_topics":
                layer = tf.einsum("ab,abc->ac", self.x, tf.transpose(self.topics, perm=[0, 2, 1]))
                layer = tf.concat([layer, tf.reshape(self.topics, (-1, self.n_topics * self.vocab_size))], axis=1)
                layer = tf.contrib.layers.batch_norm(self.transfer_fct(
                    tf.add(tf.matmul(layer, weights["h1"]), biases["b1"])))
            elif self.architecture == "standard":
                layer = tf.contrib.layers.batch_norm(self.transfer_fct(
                    tf.add(tf.matmul(self.x, weights["h1"]), biases["b1"])))
            elif self.architecture == "naive_separated":
                x_and_topics = tf.reshape(tf.concat([
                    tf.expand_dims(self.x, 1), self.topics
                    ], axis=1), (-1, (1 + self.n_topics) * self.vocab_size))
                layer_mu = tf.contrib.layers.batch_norm(self.transfer_fct(
                    tf.add(tf.matmul(x_and_topics, weights["h1_mu"]), biases["b1_mu"])))
                layer_sigma = tf.contrib.layers.batch_norm(self.transfer_fct(
                    tf.add(tf.matmul(x_and_topics, weights["h1_sigma"]), biases["b1_sigma"])))
            else:
                raise ValueError("architecture must be either 'naive' or 'template'")

            for i in range(2, self.n_hidden_layers + 1):
                if self.skip_connections and i % 2 == 0:
                    layer = layer + self.transfer_fct(
                        tf.add(tf.matmul(layer, weights["h{}".format(i)]),
                            biases["b{}".format(i)])
                    )
                else:
                    layer = self.transfer_fct(
                        tf.add(tf.matmul(layer, weights["h{}".format(i)]),
                            biases["b{}".format(i)])
                    )
                if self.use_dropout:
                    layer = tf.contrib.layers.batch_norm(tf.nn.dropout(layer, self.keep_prob))
                else:
                    layer = tf.contrib.layers.batch_norm(layer)
            if self.architecture == "naive_separated":
                z_mean = tf.contrib.layers.batch_norm(
                    tf.add(tf.matmul(layer_mu, weights["out_mean"]), biases["out_mean"])
                )
                z_log_sigma_sq = tf.contrib.layers.batch_norm(
                    tf.add(
                        tf.matmul(layer_sigma, weights["out_log_sigma"]), biases["out_log_sigma"]
                    )
                )
            else:
                z_mean = tf.contrib.layers.batch_norm(
                    tf.add(tf.matmul(layer, weights["out_mean"]), biases["out_mean"])
                )
                z_log_sigma_sq = tf.contrib.layers.batch_norm(
                    tf.add(
                        tf.matmul(layer, weights["out_log_sigma"]), biases["out_log_sigma"]
                    )
                )
            if self.tensorboard:
                for i in range(1, self.n_hidden_layers + 1):
                    if self.architecture == "naive_separated":
                        for j in ['mu', 'sigma']:
                            self.summaries.append(tf.summary.histogram(
                                "weights_h{}_{}".format(i, j), weights["h{}_{}".format(i, j)]))
                            self.summaries.append(tf.summary.histogram(
                                "biases_h{}_{}".format(i, j), biases["b{}_{}".format(i, j)]))
                    else:
                        self.summaries.append(tf.summary.histogram(
                            "weights_h{}".format(i), weights["h{}".format(i)]))
                        self.summaries.append(tf.summary.histogram(
                            "biases_h{}".format(i), biases["b{}".format(i)]))

                self.summaries.append(tf.summary.histogram("weights_out_mean", weights["out_mean"]))
                self.summaries.append(tf.summary.histogram("biases_out_mean", biases["out_mean"]))
                self.summaries.append(tf.summary.histogram("weights_out_sigma", weights["out_log_sigma"]))
                self.summaries.append(tf.summary.histogram("biases_out_sigma", biases["out_log_sigma"]))
                self.summaries.append(tf.summary.histogram("z_mean", z_mean))
                self.summaries.append(tf.summary.histogram("z_log_sigma_sq", z_log_sigma_sq))

        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, z):
        """

        :param weights:
        :param z: (batch, n_samples, n_topics)
        :return:
        """
        with tf.variable_scope("generator_network"):
            self.layer_do_0 = tf.nn.softmax(z)  # (batch, n_samples, n_topics)
            topic_weights = self.topics
            x_reconstr_means =tf.matmul(self.layer_do_0, topic_weights)  # (batch, n_samples, vocab_size)
            # x_reconstr_mean = tf.reduce_mean(x_reconstr_means, axis=1)

            if self.tensorboard:
                self.summaries.append(tf.summary.histogram("z", self.layer_do_0))

        return x_reconstr_means

    def _create_loss_optimizer(self):
        self.x_reconstr_means += 1e-10
        # The probability of the posterior of z under q
        x_expanded = tf.tile(tf.expand_dims(self.x, 1), [1, self.n_samples, 1])
        reconstr_loss = -tf.reduce_sum(
            x_expanded * tf.log(self.x_reconstr_means), axis=2
        )
        # KL Divergence
        latent_loss = 0.5 * (
            tf.reduce_sum(tf.div(self.sigma, self.var2), axis=1) +
            tf.reduce_sum(
                tf.multiply(
                    tf.div((self.mu2 - self.z_mean), self.var2),
                    (self.mu2 - self.z_mean)
                ), axis=1,
            )
            - self.h_dim
            + tf.reduce_sum(tf.log(self.var2), axis=1)
            - tf.reduce_sum(self.z_log_sigma_sq, axis=1)
        )
        # # KL annealing
        # self.cost = tf.reduce_mean(reconstr_loss) + tf.minimum(1.0, 1/50000. * tf.cast(self.global_step, tf.float32)) * tf.reduce_mean(latent_loss)
        
        # # standard training
        self.cost = tf.reduce_mean(reconstr_loss) + tf.reduce_mean(latent_loss)
        
        # beta-VAE
        # self.cost = tf.reduce_mean(reconstr_loss) + 3 * tf.reduce_mean(latent_loss)

        # # diffferent learning rates for encoder and decoder
        # enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='recognition_network')
        # dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator_network')
        # learning_rate_enc = tf.train.exponential_decay(
        #     self.starting_learning_rate, self.global_step, self.decay_steps,
        #     self.decay_rate, staircase=True)
        # learning_rate = learning_rate_enc
        # optimizer_enc = tf.train.AdamOptimizer(learning_rate_enc, beta1=0.99)
        # learning_rate_dec = tf.train.exponential_decay(
        #     self.starting_learning_rate / 100, self.global_step, self.decay_steps,
        #     self.decay_rate, staircase=True)
        # optimizer_dec = tf.train.AdamOptimizer(learning_rate_enc, beta1=0.99)
        # grad_and_vars_enc = optimizer_enc.compute_gradients(self.cost, enc_vars)
        # grad_and_vars_dec = optimizer_dec.compute_gradients(self.cost, dec_vars)
        # grad_and_vars = grad_and_vars_enc + grad_and_vars_dec
        # grads1 = [g for g, _ in grad_and_vars_enc]
        # grads2 = [g for g, _ in grad_and_vars_dec]
        # train_op1 = optimizer_enc.apply_gradients(zip(grads1, enc_vars))
        # train_op2 = optimizer_dec.apply_gradients(zip(grads2, dec_vars))
        # train_op = tf.group(train_op1, train_op2)
        # self.optimizer = train_op

        # diffferent number of steps for encoder and decoder
        train_ops = []

        enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='recognition_network')
        if self.test_lr:
            learning_rate_enc = self.lr
        elif self.custom_lr:
            learning_rate_enc, momentum = one_cycle_policy(self.starting_learning_rate, self.global_step, self.tot_epochs, self.num_batches)
            self.momentum = momentum
            self.global_step += 1
            # tf.assign_add(self.global_step, 1, name='increment')
        else:
            learning_rate_enc = tf.train.exponential_decay(
            self.starting_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, staircase=True)
        if self.use_adamw:
            weight_decay = .2
            optimizer_enc = tf.contrib.opt.AdamWOptimizer(weight_decay, learning_rate_enc)
        else:
            optimizer_enc = tf.train.AdamOptimizer(learning_rate_enc, beta1=self.momentum if self.momentum is not None else 0.99)

        dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator_network')
        if len(dec_vars):
            if self.test_lr:
                learning_rate_dec = self.lr
            elif self.custom_lr:
                learning_rate_dec = one_cycle_policy(self.starting_learning_rate, self.global_step)
                self.global_step += 1
            else:
                learning_rate_dec = tf.train.exponential_decay(
                    self.starting_learning_rate, self.global_step, self.decay_steps,
                    self.decay_rate, staircase=True)
            if self.use_adamw:
                weight_decay = .2
                optimizer_dec = tf.contrib.opt.AdamWOptimizer(weight_decay, learning_rate_dec)
            else:
                optimizer_dec = tf.train.AdamOptimizer(learning_rate_dec, beta1=0.99)
            grad_and_vars_dec = optimizer_dec.compute_gradients(self.cost, dec_vars)
            train_op_dec = optimizer_dec.apply_gradients(grad_and_vars_dec)
            train_ops.append(train_op_dec)

        enc_gv_collection = []
        for i in range(self.n_steps_enc):
            grad_and_vars_enc = optimizer_enc.compute_gradients(self.cost, enc_vars)
            enc_gv_collection.append(grad_and_vars_enc)
            if i == self.n_steps_enc - 1 and not self.custom_lr:
                train_ops.append(optimizer_enc.apply_gradients(grad_and_vars_enc, global_step=self.global_step))
            else:
                train_ops.append(optimizer_enc.apply_gradients(grad_and_vars_enc))
        train_op = tf.group(train_ops)
        self.optimizer = train_op

        # # typical learning rate
        # learning_rate = tf.train.exponential_decay(
        #     self.starting_learning_rate, self.global_step, self.decay_steps,
        #     self.decay_rate, staircase=True)
        # optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.99)
        # grad_and_vars = optimizer.compute_gradients(loss=self.cost)
        # # Passing global_step to minimize() will increment it at each step.
        # self.optimizer = optimizer.apply_gradients(grad_and_vars, global_step=self.global_step)

        if self.tensorboard:
            with tf.name_scope("performance"):
                self.summaries.append(tf.summary.scalar("loss", self.cost))
                self.summaries.append(tf.summary.scalar(
                    "latent_loss", tf.reduce_mean(latent_loss)))
                self.summaries.append(tf.summary.scalar(
                    "reconstruction_loss", tf.reduce_mean(reconstr_loss)))
                self.summaries.append(tf.summary.scalar("learning_rate_enc", learning_rate_enc))
                if self.momentum is not None:
                    self.summaries.append(tf.summary.scalar("momentum_enc", self.momentum))
                if dec_vars:
                    self.summaries.append(tf.summary.scalar("learning_rate_dec", learning_rate_dec))
            # TODO: add the accumulated gradients from the encoder
            if dec_vars:
                for gradient, variable in grad_and_vars_dec:
                    variable_name = variable.name.replace(':', '_')
                    self.summaries.append(tf.summary.histogram("dec_gradients/" + variable_name, l2_norm(gradient)))
                    self.summaries.append(tf.summary.histogram("dec_variables/" + variable_name, l2_norm(variable)))

    def partial_fit(self, X_and_topics, learning_rate=None):
        X, topics = unzip_X_and_topics(X_and_topics)
        logger.info('Finished unzipping X and topics')
        if self.test_lr:
            opt, cost = self.sess.run(
            (self.optimizer, self.cost),
            feed_dict={self.x: X, self.topics: topics, self.keep_prob: 0.75, self.lr: learning_rate},
        )
        else:
            opt, cost = self.sess.run(
                (self.optimizer, self.cost),
                feed_dict={self.x: X, self.topics: topics, self.keep_prob: 0.75},
            )
        return cost

    def recreate_input(self, X_and_topics):
        X, topics = unzip_X_and_topics(X_and_topics)
        output, z_mean, z = self.sess.run(
            (self.x_reconstr_mean, self.z_mean, self.z),
            feed_dict={self.x: X, self.topics: topics, self.keep_prob: 1.0}
        )
        return output, z_mean, z

    def get_latents(self, X_and_topics):
        X, topics = unzip_X_and_topics(X_and_topics)
        z_mean, z_log_sigma_sq = self.sess.run(
            (self.z_mean, self.z_log_sigma_sq), feed_dict={self.x: X, self.topics: topics, self.keep_prob: 1.0}
        )
        return z_mean, z_log_sigma_sq

    def test(self, X_and_topics):
        X, topics = unzip_X_and_topics(X_and_topics)
        """ Best for a single input of shape (dim,). Note the np.expand_dims() """
        cost = self.sess.run(
            (self.cost),
            feed_dict={self.x: np.expand_dims(X, axis=0), self.topics: np.expand_dims(topics, axis=0), self.keep_prob: 1.0},
        )
        return cost

    def evaluate(self, X_and_topics):
        X, topics = unzip_X_and_topics(X_and_topics)
        cost = self.sess.run(
            (self.cost),
            feed_dict={self.x: X, self.topics: topics, self.keep_prob: 1.0},
        )
        return cost

    def doc_prop(self, X_and_topics):
        X, topics = unzip_X_and_topics(X_and_topics)
        doc_topic_proportions = self.sess.run(self.z_mean,
                                              feed_dict={self.x: X, self.topics: topics, self.keep_prob: 1.0}
                                              )
        return doc_topic_proportions

    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.n_topics)
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.z: z_mu})

    def save(self):
        if not os.path.exists(self.results_dir):
            os.system('mkdir -p ' + self.results_dir)
        file_name = '_'.join([self.model_name, str(self.n_hidden_layers), str(self.n_hidden_units)]) + '.h5'
        h5f = h5py.File(os.path.join(self.results_dir, file_name), 'w')
        for i in range(self.n_hidden_layers):
            if self.architecture == 'naive_separated':
                for j in ['mu', 'sigma']:
                    weights = self.sess.graph.get_tensor_by_name('recognition_network/h{}_{}:0'.format(i + 1, j)).eval(session=self.sess)
                    biases = self.sess.graph.get_tensor_by_name('recognition_network/b{}_{}:0'.format(i + 1, j)).eval(session=self.sess)
                    h5f.create_dataset("weights_{}_{}".format(i + 1, j) , data=weights)
                    h5f.create_dataset("biases_{}_{}".format(i + 1, j), data=biases)
                    # TODO: fix the hack here
                    if j == 'mu':
                        beta = self.sess.graph.get_tensor_by_name('recognition_network/BatchNorm/beta:0').eval(session=self.sess)
                        running_mean = self.sess.graph.get_tensor_by_name('recognition_network/BatchNorm/moving_mean:0').eval(session=self.sess)
                        running_var = self.sess.graph.get_tensor_by_name('recognition_network/BatchNorm/moving_variance:0').eval(session=self.sess)
                    elif j == 'sigma':
                        beta = self.sess.graph.get_tensor_by_name('recognition_network/BatchNorm_{}/beta:0'.format(1)).eval(session=self.sess)
                        running_mean = self.sess.graph.get_tensor_by_name('recognition_network/BatchNorm_{}/moving_mean:0'.format(1)).eval(session=self.sess)
                        running_var = self.sess.graph.get_tensor_by_name('recognition_network/BatchNorm_{}/moving_variance:0'.format(1)).eval(session=self.sess)
                    h5f.create_dataset("beta_{}_{}".format(i + 1, j), data=beta)
                    h5f.create_dataset("running_mean_{}_{}".format(i + 1, j), data=running_mean)
                    h5f.create_dataset("running_var_{}_{}".format(i + 1, j), data=running_var)

            else:
                weights = self.sess.graph.get_tensor_by_name('recognition_network/h{}:0'.format(i + 1)).eval(session=self.sess)
                biases = self.sess.graph.get_tensor_by_name('recognition_network/b{}:0'.format(i + 1)).eval(session=self.sess)
                h5f.create_dataset("weights_{}".format(i + 1) , data=weights)
                h5f.create_dataset("biases_{}".format(i + 1), data=biases)
                if i == 0:
                    beta = self.sess.graph.get_tensor_by_name('recognition_network/BatchNorm/beta:0').eval(session=self.sess)
                    running_mean = self.sess.graph.get_tensor_by_name('recognition_network/BatchNorm/moving_mean:0').eval(session=self.sess)
                    running_var = self.sess.graph.get_tensor_by_name('recognition_network/BatchNorm/moving_variance:0').eval(session=self.sess)
                else:
                    beta = self.sess.graph.get_tensor_by_name('recognition_network/BatchNorm_{}/beta:0'.format(i)).eval(session=self.sess)
                    running_mean = self.sess.graph.get_tensor_by_name('recognition_network/BatchNorm_{}/moving_mean:0'.format(i)).eval(session=self.sess)
                    running_var = self.sess.graph.get_tensor_by_name('recognition_network/BatchNorm_{}/moving_variance:0'.format(i)).eval(session=self.sess)
                h5f.create_dataset("beta_{}".format(i + 1), data=beta)
                h5f.create_dataset("running_mean_{}".format(i + 1), data=running_mean)
                h5f.create_dataset("running_var_{}".format(i + 1), data=running_var)

        # out_mean and out_sigma
        weights_out_mean = self.sess.graph.get_tensor_by_name('recognition_network/out_mean:0').eval(session=self.sess)
        weights_out_log_sigma = self.sess.graph.get_tensor_by_name('recognition_network/out_log_sigma:0').eval(session=self.sess)
        biases_out_mean = self.sess.graph.get_tensor_by_name('recognition_network/out_mean_b:0').eval(session=self.sess)
        biases_out_log_sigma = self.sess.graph.get_tensor_by_name('recognition_network/out_log_sigma_b:0').eval(session=self.sess)
        if self.architecture == 'naive_separated':
            offset = 1
        else:
            offset = 0
        beta_out_mean = self.sess.graph.get_tensor_by_name('recognition_network/BatchNorm_{}/beta:0'.format(self.n_hidden_layers + offset)).eval(session=self.sess)
        beta_out_log_sigma = self.sess.graph.get_tensor_by_name('recognition_network/BatchNorm_{}/beta:0'.format(self.n_hidden_layers + offset + 1)).eval(session=self.sess)
        running_mean_out_mean = self.sess.graph.get_tensor_by_name('recognition_network/BatchNorm_{}/moving_mean:0'.format(self.n_hidden_layers + offset)).eval(
            session=self.sess)
        running_mean_out_log_sigma = self.sess.graph.get_tensor_by_name('recognition_network/BatchNorm_{}/moving_mean:0'.format(self.n_hidden_layers + offset + 1)).eval(
            session=self.sess)
        running_var_out_mean = self.sess.graph.get_tensor_by_name('recognition_network/BatchNorm_{}/moving_variance:0'.format(self.n_hidden_layers + offset)).eval(
            session=self.sess)
        running_var_out_log_sigma = self.sess.graph.get_tensor_by_name('recognition_network/BatchNorm_{}/moving_variance:0'.format(self.n_hidden_layers + offset + 1)).eval(
            session=self.sess)


        h5f.create_dataset("weights_out_mean", data=weights_out_mean)
        h5f.create_dataset("weights_out_log_sigma", data=weights_out_log_sigma)
        h5f.create_dataset("biases_out_mean", data=biases_out_mean)
        h5f.create_dataset("biases_out_log_sigma", data=biases_out_log_sigma)
        h5f.create_dataset("beta_out_mean", data=beta_out_mean)
        h5f.create_dataset("beta_out_log_sigma", data=beta_out_log_sigma)
        h5f.create_dataset("running_mean_out_mean", data=running_mean_out_mean)
        h5f.create_dataset("running_mean_out_log_sigma", data=running_mean_out_log_sigma)
        h5f.create_dataset("running_var_out_mean", data=running_var_out_mean)
        h5f.create_dataset("running_var_out_log_sigma", data=running_var_out_log_sigma)

        # scale
        scale = self.sess.graph.get_tensor_by_name('recognition_network/scale:0').eval(session=self.sess)
        h5f.create_dataset("scale", data=scale)

        h5f.close()


class MLP(nn.Module):
    def __init__(self, n_input_units, n_output_units):
        super(MLP, self).__init__()
        self.fc = nn.Linear(n_input_units, n_output_units)
        self.softplus = nn.Softplus()
        self.bn = nn.BatchNorm1d(n_output_units)

    def forward(self, x):
        return self.bn(self.softplus(self.fc(x)))

class MLP_with_skip(MLP):
    def forward(self, x):
        return self.bn(x + self.softplus(self.fc(x)))

class Encoder(nn.Module):
    def __init__(self, n_hidden_units, n_hidden_layers, architecture, n_topics=4, vocab_size=9, use_scale=False, skip_connections=False):
        super(Encoder, self).__init__()
        self.n_hidden_layers = n_hidden_layers
        self.vocab_size = vocab_size
        self.n_topics = n_topics
        # setup the non-linearities
        self.softplus = nn.Softplus()
        self.skip_connections = skip_connections
        # encoder Linear layers
        modules = []
        self.architecture = architecture
        if architecture == 'naive' or architecture == 'naive_separated':
            modules.append(MLP((1 + n_topics) * vocab_size, n_hidden_units))
        elif architecture == 'template':
            modules.append(MLP(n_topics, n_hidden_units))
        elif architecture == 'template_plus_topics':
            modules.append(MLP(n_topics * (1 + vocab_size), n_hidden_units))
        elif architecture == 'standard':
            modules.append(MLP(vocab_size, n_hidden_units))
        else:
            raise ValueError('Invalid architecture')
        for i in range(self.n_hidden_layers - 1):
            if self.skip_connections and i % 2 == 0:
                modules.append(MLP_with_skip(n_hidden_units, n_hidden_units))
            else:
                modules.append(MLP(n_hidden_units, n_hidden_units))

        if architecture == 'naive_separated':
            self.enc_layers_mu = nn.Sequential(*modules)
            self.enc_layers_sigma = nn.Sequential(*modules)
        else:
            self.enc_layers = nn.Sequential(*modules)
        self.fcmu = nn.Linear(n_hidden_units, n_topics)
        self.fcsigma = nn.Linear(n_hidden_units, n_topics)
        self.bnmu = nn.BatchNorm1d(n_topics)
        self.bnsigma = nn.BatchNorm1d(n_topics)
        self.use_scale = use_scale
        if self.use_scale:
            self.scale = nn.Parameter(torch.ones([]))

    def forward(self, x, topics):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.reshape(-1, self.vocab_size)
        if self.architecture == 'naive':
            x_and_topics = torch.cat((x, topics.reshape(-1, self.n_topics * self.vocab_size)), dim=1)
            # then return a mean vector and a (positive) square root covariance
            # each of size batch_size x n_topics
            z_loc = self.bnmu(self.fcmu(self.enc_layers(x_and_topics)))
            z_scale = torch.sqrt(torch.exp(self.bnsigma(self.fcsigma(self.enc_layers(x_and_topics)))))
        elif self.architecture == 'template':
            x_and_topics = torch.einsum("ab,abc->ac", (x, torch.transpose(topics, 1, 2)))
            z_loc = self.bnmu(self.fcmu(self.enc_layers(x_and_topics)))
            z_scale = torch.sqrt(torch.exp(self.bnsigma(self.fcsigma(self.enc_layers(x_and_topics)))))
        elif self.architecture == 'template_plus_topics':
            x_and_topics = torch.einsum("ab,abc->ac", (x, torch.transpose(topics, 1, 2)))
            x_and_topics = torch.cat((x_and_topics, topics.reshape(-1, self.n_topics * self.vocab_size)), dim=1)
            z_loc = self.bnmu(self.fcmu(self.enc_layers(x_and_topics)))
            z_scale = torch.sqrt(torch.exp(self.bnsigma(self.fcsigma(self.enc_layers(x_and_topics)))))
        elif self.architecture == 'standard':
            z_loc = self.bnmu(self.fcmu(self.enc_layers(x)))
            z_scale = torch.sqrt(torch.exp(self.bnsigma(self.fcsigma(self.enc_layers(x)))))
        elif self.architecture == 'naive_separated':
            x_and_topics = torch.cat((x, topics.reshape(-1, self.n_topics * self.vocab_size)), dim=1)
            # then return a mean vector and a (positive) square root covariance
            # each of size batch_size x n_topics
            z_loc = self.bnmu(self.fcmu(self.enc_layers_mu(x_and_topics)))
            z_scale = torch.sqrt(torch.exp(self.bnsigma(self.fcsigma(self.enc_layers_sigma(x_and_topics)))))
        else:
            raise ValueError('Invalid architecture')
        if self.use_scale:
            z_loc = torch.mul(self.scale, z_loc)
        return z_loc, z_scale


class Decoder(nn.Module):
    def __init__(self, use_scale=False):
        super(Decoder, self).__init__()
        self.use_scale = use_scale
        if self.use_scale:
            self.scale = nn.Parameter(torch.ones([]), requires_grad=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, z, topics):
        if self.use_scale:
            print(self.scale)
            z = torch.mul(autograd.Variable(self.scale), z)
        word_probs = torch.bmm(self.softmax(z).unsqueeze(1), topics)
        return torch.squeeze(word_probs, 1)


class VAE_pyro(nn.Module):
    def __init__(self, n_hidden_units=100, n_hidden_layers=2, model_name=None, results_dir=None,
                 alpha=.1, vocab_size=9, n_topics=4, use_cuda=False, architecture='naive', scale_type='sample', skip_connections=False, **kwargs):
        super(VAE_pyro, self).__init__()

        # create the encoder and decoder networks
        self.scale_type = scale_type
        if self.scale_type == 'sample':
            self.encoder = Encoder(n_hidden_units, n_hidden_layers, architecture, n_topics=n_topics, vocab_size=vocab_size, use_scale=False, skip_connections=skip_connections)
            self.decoder = Decoder(use_scale=True)
        elif self.scale_type == 'mean':
            self.encoder = Encoder(n_hidden_units, n_hidden_layers, architecture, n_topics=n_topics, vocab_size=vocab_size, use_scale=True, skip_connections=skip_connections)
            self.decoder = Decoder(use_scale=False)
        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.n_topics = n_topics
        alpha_vec = alpha * np.ones((1, n_topics)).astype(np.float32)
        self.z_loc = torch.from_numpy((np.log(alpha_vec).T - np.mean(np.log(alpha_vec), 1)).T)
        self.z_scale = torch.from_numpy((
            ((1.0 / alpha_vec) * (1 - (2.0 / n_topics))).T +
            (1.0 / (n_topics * n_topics)) * np.sum(1.0 / alpha_vec, 1)
        ).T)
        self.alpha = alpha * np.ones(n_topics).astype(np.float32)

        self.n_topics = n_topics
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units
        self.model_name = model_name
        self.results_dir = results_dir
        self.architecture = architecture

    # define the model p(x|z)p(z)
    def model(self, x, topics):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z = pyro.sample("latent",
                            dist.Normal(self.z_loc, self.z_scale).to_event(1))
            word_probs = self.decoder.forward(z, topics)
            return pyro.sample("doc_words",
                        dist.Multinomial(probs=word_probs),
                        obs=x)
                        # obs=x.reshape(-1, VOCAB_SIZE))

    # define the model p(x|z)p(z)
    def lda_model(self, x, topics):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z = pyro.sample("latent",
                            dist.Dirichlet(torch.from_numpy(self.alpha)))
            word_probs = self.decoder.forward(z, topics)
            return pyro.sample("doc_words",
                        dist.Multinomial(probs=word_probs),
                        obs=x)


    def encoder_guide(self, x, topics):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x, topics)
            # sample the latent code z
            pyro.sample("latent",
                        dist.Normal(z_loc, z_scale).to_event(1))
                        # dist.LogisticNormal(z_loc, z_scale).to_event(1))

    def mean_field_guide(self, x, topics):
        with pyro.plate("data", x.shape[0]):
            # z_loc = pyro.param("z_loc", self.z_loc)
            # z_loc = pyro.param("z_loc", self.z_loc.repeat(x.shape[0], 1))
            z_loc = pyro.param("z_loc", x.new_zeros(torch.Size((x.shape[0], self.n_topics))))

            # z_scale = pyro.param("z_scale", self.z_scale)
            # z_scale = pyro.param("z_scale", self.z_scale.repeat(x.shape[0], 1))
            z_scale = pyro.param("z_scale", x.new_ones(torch.Size((x.shape[0], self.n_topics))))

            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def map_guide(self, x, topics):
        with pyro.plate("data", x.shape[0]):
            z_loc = pyro.param("z_loc", self.z_loc)
            pyro.sample("latent", dist.Delta(z_loc).to_event(1))

    def reconstruct_with_vae_map(self, x, topics, sample_z=False):
        # encode image x
        z_loc, z_scale = self.encoder(x, topics)
        if sample_z:
            # sample in latent space
            z_loc = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        word_probs = self.decoder(z_loc, topics)
        return word_probs

    def load(self):
        state_dict = {}
        file_name = '_'.join([self.model_name, str(self.n_hidden_layers), str(self.n_hidden_units)]) + '.h5'
        h5f = h5py.File(os.path.join(self.results_dir, file_name), 'r')
        for i in range(self.n_hidden_layers):
            if self.architecture == 'naive_separated':
                for j in ['mu', 'sigma']:
                    state_dict['encoder.enc_layers_{}.{}.fc.weight'.format(j, i)] = torch.from_numpy(h5f['weights_{}_{}'.format(i + 1, j)][()]).t()
                    state_dict['encoder.enc_layers_{}.{}.fc.bias'.format(j, i)] = torch.from_numpy(h5f['biases_{}_{}'.format(i + 1, j)][()])
                    state_dict['encoder.enc_layers_{}.{}.bn.bias'.format(j, i)] = torch.from_numpy(h5f['beta_{}_{}'.format(i + 1, j)][()])
                    state_dict['encoder.enc_layers_{}.{}.bn.weight'.format(j, i)] = torch.ones(self.n_hidden_units)
                    state_dict['encoder.enc_layers_{}.{}.bn.running_mean'.format(j, i)] = torch.from_numpy(h5f['running_mean_{}_{}'.format(i + 1, j)][()])
                    state_dict['encoder.enc_layers_{}.{}.bn.running_var'.format(j, i)] = torch.from_numpy(h5f['running_var_{}_{}'.format(i + 1, j)][()])
            else:
                state_dict['encoder.enc_layers.{}.fc.weight'.format(i)] = torch.from_numpy(h5f['weights_{}'.format(i + 1)][()]).t()
                state_dict['encoder.enc_layers.{}.fc.bias'.format(i)] = torch.from_numpy(h5f['biases_{}'.format(i + 1)][()])
                state_dict['encoder.enc_layers.{}.bn.bias'.format(i)] = torch.from_numpy(h5f['beta_{}'.format(i + 1)][()])
                state_dict['encoder.enc_layers.{}.bn.weight'.format(i)] = torch.ones(self.n_hidden_units)
                state_dict['encoder.enc_layers.{}.bn.running_mean'.format(i)] = torch.from_numpy(h5f['running_mean_{}'.format(i + 1)][()])
                state_dict['encoder.enc_layers.{}.bn.running_var'.format(i)] = torch.from_numpy(h5f['running_var_{}'.format(i + 1)][()])

        state_dict['encoder.fcmu.weight'] = torch.from_numpy(h5f['weights_out_mean'][()]).t()
        state_dict['encoder.fcsigma.weight'] = torch.from_numpy(h5f['weights_out_log_sigma'][()]).t()
        state_dict['encoder.fcmu.bias'] = torch.from_numpy(h5f['biases_out_mean'][()])
        state_dict['encoder.fcsigma.bias'] = torch.from_numpy(h5f['biases_out_log_sigma'][()])

        state_dict['encoder.bnmu.bias'] = torch.from_numpy(h5f['beta_out_mean'][()])
        state_dict['encoder.bnsigma.bias'] = torch.from_numpy(h5f['beta_out_log_sigma'][()])
        state_dict['encoder.bnmu.weight'] = torch.ones(self.n_topics)
        state_dict['encoder.bnsigma.weight'] = torch.ones(self.n_topics)
        state_dict['encoder.bnmu.running_mean'] = torch.from_numpy(h5f['running_mean_out_mean'][()])
        state_dict['encoder.bnsigma.running_mean'] = torch.from_numpy(h5f['running_mean_out_log_sigma'][()])
        state_dict['encoder.bnmu.running_var'] = torch.from_numpy(h5f['running_var_out_mean'][()])
        state_dict['encoder.bnsigma.running_var'] = torch.from_numpy(h5f['running_var_out_log_sigma'][()])

        if self.scale_type == 'sample':
            state_dict['decoder.scale'] = torch.from_numpy(np.array(h5f['scale'][()]))
        elif self.scale_type == 'mean':
            state_dict['encoder.scale'] = torch.from_numpy(np.array(h5f['scale'][()]))
        h5f.close()

        return state_dict
