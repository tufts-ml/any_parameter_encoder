import os
import h5py
import numpy as np
import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist

import numpy as np
import tensorflow as tf
from functools import partial


l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
cwd = os.getcwd()


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
            topic_init=None,
            topic_trainable=True,
            enc_topic_init=None,
            enc_topic_trainable=True,
            scale_trainable=False,
            transfer_fct=tf.nn.softplus,
            starting_learning_rate=0.002,
            decay_steps=1000,
            decay_rate=.9,
            n_samples=1,
            batch_size=200,
            alpha=1,
            n_steps_enc=1,
            tensorboard=False,
            **kwargs
    ):
        self.n_hidden_units = n_hidden_units
        self.n_hidden_layers = n_hidden_layers
        self.n_topics = n_topics
        self.model_name = model_name
        self.results_dir = results_dir
        self.vocab_size = vocab_size
        self.topic_init = topic_init
        self.topic_trainable = topic_trainable
        self.enc_topic_init = enc_topic_init
        self.enc_topic_trainable = enc_topic_trainable
        self.scale_trainable = scale_trainable
        self.transfer_fct = transfer_fct
        self.starting_learning_rate = starting_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.n_steps_enc = n_steps_enc
        self.tensorboard = tensorboard
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.summaries = []
        """----------------Inputs----------------"""
        self.x = tf.placeholder(tf.float32, [None, self.vocab_size], name='x')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

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
        self._create_network()
        self._create_loss_optimizer()
        init = tf.global_variables_initializer()
        # self.sess = tf.InteractiveSession()
        self.sess = tf.Session()
        self.sess.run(init)

    def _create_network(self):
        self.network_weights = self._initialize_weights()

        self.z_mean, self.z_log_sigma_sq = self._recognition_network(
            self.network_weights["weights_recog"], self.network_weights["biases_recog"]
        )
        eps = tf.random_normal((self.n_samples, self.n_topics), 0, 1, dtype=tf.float32)
        z = tf.add(
            tf.expand_dims(self.z_mean, axis=1), tf.multiply(tf.sqrt(tf.exp(tf.expand_dims(self.z_log_sigma_sq, axis=1))), eps)
        )
        z = tf.multiply(self.scale, z)
        self.z = tf.reduce_mean(z, axis=1)
        self.sigma = tf.exp(self.z_log_sigma_sq)
        # generator = partial(self._generator_network, self.network_weights['weights_gener'])
        # self.x_reconstr_mean = tf.reduce_mean(tf.map_fn(generator, self.z), axis=1)
        self.x_reconstr_means = self._generator_network(self.network_weights['weights_gener'], z)
        self.x_reconstr_mean = tf.reduce_mean(self.x_reconstr_means, axis=1)

    def _initialize_weights(self):
        all_weights = dict()
        with tf.variable_scope("recognition_network"):
            all_weights["weights_recog"] = {
                "out_mean": tf.get_variable("out_mean", [self.n_hidden_units, self.n_topics]),
                "out_log_sigma": tf.get_variable("out_log_sigma", [self.n_hidden_units, self.n_topics]),
            }

            all_weights["biases_recog"] = {
                "out_mean": tf.get_variable("out_mean_b", [self.n_topics], initializer=tf.zeros_initializer()),
                "out_log_sigma": tf.get_variable("out_log_sigma_b", [self.n_topics], initializer=tf.zeros_initializer()),
            }

            # add more layers
            if self.n_hidden_layers > 1:
                for i in range(2, self.n_hidden_layers + 1):
                    if not self.enc_topic_init:
                        all_weights["weights_recog"]["h{}".format(i)] = tf.get_variable(
                            "h{}".format(i), [self.n_hidden_units, self.n_hidden_units])
                        all_weights["biases_recog"]["b{}".format(i)] = tf.get_variable(
                            "b{}".format(i), [self.n_hidden_units], initializer=tf.zeros_initializer())

            if self.enc_topic_init:
                # if self.n_hidden_units != self.n_topics:
                #     raise ValueError("If initializing encoder topics, must enforce n_hidden_units == n_topics.")
                toy_bars = tf.transpose(tf.convert_to_tensor(np.load(
                    os.path.join(cwd, self.topic_init))))
                all_weights["weights_recog"].update({
                    "h1": tf.get_variable(
                        "h1", initializer=toy_bars, trainable=self.enc_topic_trainable)})
                all_weights["weights_recog"].update({
                    "b1": tf.get_variable("b1", [self.n_topics], initializer=tf.zeros_initializer()),
                })
            else:
                all_weights["weights_recog"].update({
                    "h1": tf.get_variable("h1", [self.vocab_size, self.n_hidden_units]),
                    "b1": tf.get_variable("b1", [self.n_hidden_units], initializer=tf.zeros_initializer())})

            self.scale = tf.Variable(tf.ones([]), name="scale", trainable=self.scale_trainable)

        # initialize the generative model
        with tf.variable_scope("generator_network"):
            if self.topic_init:
                toy_bars = tf.convert_to_tensor(np.load(
                    os.path.join(cwd, self.topic_init)))
                all_weights["weights_gener"] = {
                    "g1": tf.get_variable(
                        "g1", initializer=toy_bars, trainable=self.topic_trainable)}
            else:
                all_weights["weights_gener"] = {
                    "g1": tf.get_variable(
                        "g1", [self.n_topics, self.vocab_size],
                        initializer=tf.contrib.layers.xavier_initializer())
                }

        return all_weights

    def _recognition_network(self, weights, biases):
        with tf.variable_scope("recognition_network"):
            layer = self.transfer_fct(
                tf.add(tf.matmul(self.x, weights["h1"]), biases["b1"]))

            for i in range(2, self.n_hidden_layers + 1):
                layer = self.transfer_fct(
                    tf.add(tf.matmul(layer, weights["h{}".format(i)]),
                           biases["b{}".format(i)])
                )
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

    def _generator_network(self, weights, z):
        """

        :param weights:
        :param z: (batch, n_samples, n_topics)
        :return:
        """
        with tf.variable_scope("generator_network"):
            self.layer_do_0 = tf.nn.softmax(z)
            topic_weights = tf.tile(tf.expand_dims(tf.nn.softmax(weights["g1"]), 0), [tf.shape(z)[0], 1, 1])
            x_reconstr_means =tf.matmul(self.layer_do_0, topic_weights)  # (batch, n_samples, vocab_size)
            # x_reconstr_mean = tf.reduce_mean(x_reconstr_means, axis=1)

            if self.tensorboard:
                self.summaries.append(tf.summary.histogram("weights", weights["g1"]))
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
        enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='recognition_network')
        dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator_network')
        learning_rate_enc = tf.train.exponential_decay(
            self.starting_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, staircase=True)
        learning_rate = learning_rate_enc
        optimizer_enc = tf.train.AdamOptimizer(learning_rate_enc, beta1=0.99)
        learning_rate_dec = tf.train.exponential_decay(
            self.starting_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, staircase=True)
        optimizer_dec = tf.train.AdamOptimizer(learning_rate_enc, beta1=0.99)
        grad_and_vars_dec = optimizer_dec.compute_gradients(self.cost, dec_vars)
        train_op_dec = optimizer_dec.apply_gradients(grad_and_vars_dec)
        train_ops = [train_op_dec]
        enc_gv_collection = []
        for i in range(self.n_steps_enc):
            grad_and_vars_enc = optimizer_enc.compute_gradients(self.cost, enc_vars)
            enc_gv_collection.append(grad_and_vars_enc)
            train_ops.append(optimizer_enc.apply_gradients(grad_and_vars_enc))
        train_op = tf.group(train_ops)
        self.optimizer = train_op
        # TODO: add the accumulated gradients from the encoder
        grad_and_vars = grad_and_vars_dec

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
                self.summaries.append(tf.summary.scalar("learning_rate", learning_rate))
            for gradient, variable in grad_and_vars:
                variable_name = variable.name.replace(':', '_')
                self.summaries.append(tf.summary.histogram("gradients/" + variable_name, l2_norm(gradient)))
                self.summaries.append(tf.summary.histogram("variables/" + variable_name, l2_norm(variable)))

    def partial_fit(self, X):
        opt, cost = self.sess.run(
            (self.optimizer, self.cost),
            feed_dict={self.x: X, self.keep_prob: 0.75},
        )
        return cost

    def recreate_input(self, X):
        output, z_mean, z = self.sess.run(
            (self.x_reconstr_mean, self.z_mean, self.z),
            feed_dict={self.x: X, self.keep_prob: 1.0}
        )
        return output, z_mean, z

    def get_latents(self, X):
        z_mean, z_log_sigma_sq = self.sess.run(
            (self.z_mean, self.z_log_sigma_sq), feed_dict={self.x: X, self.keep_prob: 1.0}
        )
        return z_mean, z_log_sigma_sq

    def test(self, X):
        """ Best for a single input of shape (dim,). Note the np.expand_dims() """
        cost = self.sess.run(
            (self.cost),
            feed_dict={self.x: np.expand_dims(X, axis=0), self.keep_prob: 1.0},
        )
        return cost

    def evaluate(self, X):
        cost = self.sess.run(
            (self.cost),
            feed_dict={self.x: X, self.keep_prob: 1.0},
        )
        return cost

    def topic_prop(self, X):
        topic_word_proportions = self.sess.run(
            (self.network_weights["weights_gener"]["g1"]),
            feed_dict={self.x: X, self.keep_prob: 1.0}
        )
        return topic_word_proportions

    def doc_prop(self, X):
        doc_topic_proportions = self.sess.run(self.z_mean,
                                              feed_dict={self.x: X, self.keep_prob: 1.0}
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
            weights = self.sess.graph.get_tensor_by_name('recognition_network/h{}:0'.format(i + 1)).eval(session=self.sess)
            biases = self.sess.graph.get_tensor_by_name('recognition_network/b{}:0'.format(i + 1)).eval(session=self.sess)
            h5f.create_dataset("weights_{}".format(i + 1) , data=weights)
            h5f.create_dataset("biases_{}".format(i + 1), data=biases)

        # out_mean and out_sigma
        weights_out_mean = self.sess.graph.get_tensor_by_name('recognition_network/out_mean:0').eval(session=self.sess)
        weights_out_log_sigma = self.sess.graph.get_tensor_by_name('recognition_network/out_log_sigma:0').eval(session=self.sess)
        biases_out_mean = self.sess.graph.get_tensor_by_name('recognition_network/out_mean_b:0').eval(session=self.sess)
        biases_out_log_sigma = self.sess.graph.get_tensor_by_name('recognition_network/out_log_sigma_b:0').eval(session=self.sess)
        beta_out_mean = self.sess.graph.get_tensor_by_name('recognition_network/BatchNorm/beta:0').eval(session=self.sess)
        beta_out_log_sigma = self.sess.graph.get_tensor_by_name('recognition_network/BatchNorm_1/beta:0').eval(session=self.sess)
        running_mean_out_mean = self.sess.graph.get_tensor_by_name('recognition_network/BatchNorm/moving_mean:0').eval(
            session=self.sess)
        running_mean_out_log_sigma = self.sess.graph.get_tensor_by_name('recognition_network/BatchNorm_1/moving_mean:0').eval(
            session=self.sess)
        running_var_out_mean = self.sess.graph.get_tensor_by_name('recognition_network/BatchNorm/moving_variance:0').eval(
            session=self.sess)
        running_var_out_log_sigma = self.sess.graph.get_tensor_by_name('recognition_network/BatchNorm_1/moving_variance:0').eval(
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

        # topics
        topics = self.sess.graph.get_tensor_by_name('generator_network/g1:0').eval(session=self.sess)
        h5f.create_dataset("topics", data=topics)

        h5f.close()


class MLP(nn.Module):
    def __init__(self, n_input_units, n_output_units):
        super(MLP, self).__init__()
        self.fc = nn.Linear(n_input_units, n_output_units)
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.softplus(self.fc(x))

class Encoder(nn.Module):
    def __init__(self, n_hidden_units, n_hidden_layers, n_topics=4, vocab_size=9):
        super(Encoder, self).__init__()
        self.n_hidden_layers = n_hidden_layers
        self.vocab_size = vocab_size
        # setup the non-linearities
        self.softplus = nn.Softplus()
        # encoder Linear layers
        modules = []
        modules.append(MLP(vocab_size, n_hidden_units))
        for i in range(self.n_hidden_layers - 1):
            modules.append(MLP(n_hidden_units, n_hidden_units))

        self.enc_layers = nn.Sequential(*modules)
        self.fcmu = nn.Linear(n_hidden_units, n_topics)
        self.fcsigma = nn.Linear(n_hidden_units, n_topics)
        self.bnmu = nn.BatchNorm1d(n_topics)
        self.bnsigma = nn.BatchNorm1d(n_topics)
        self.scale = nn.Parameter(torch.ones([]))

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.reshape(-1, self.vocab_size)
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x n_topics
        z_loc = torch.mul(self.scale, self.bnmu(self.fcmu(self.enc_layers(x))))
        z_scale = torch.exp(self.bnsigma(self.fcsigma(self.enc_layers(x))))
        return z_loc, z_scale


class Decoder(nn.Module):
    def __init__(self, n_topics, vocab_size, topic_init, topic_trainable):
        super(Decoder, self).__init__()
        if topic_init:
            # the topics are already in inverse_softmax form
            topics = np.load(topic_init)
            self.topics = torch.tensor(topics, requires_grad=topic_trainable)
        else:
            self.topics = torch.nn.Parameter(torch.randn(n_topics, vocab_size))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, z):
        word_probs = torch.mm(self.softmax(z), self.softmax(self.topics))
        return word_probs


class VAE_pyro(nn.Module):
    def __init__(self, n_hidden_units=100, n_hidden_layers=2, model_name=None, results_dir=None, topic_init=None, topic_trainable=True,
                 alpha=.1, vocab_size=9, n_topics=4, use_cuda=False, **kwargs):
        super(VAE_pyro, self).__init__()
        print(n_hidden_units)
        print(n_hidden_layers)
        # create the encoder and decoder networks
        self.encoder = Encoder(n_hidden_units, n_hidden_layers, n_topics=n_topics, vocab_size=vocab_size)
        self.decoder = Decoder(n_topics, vocab_size, topic_init, topic_trainable)

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
        self.topic_init = topic_init

    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z = pyro.sample("latent",
                            dist.Normal(self.z_loc, self.z_scale).to_event(1))
            word_probs = self.decoder.forward(z)
            return pyro.sample("doc_words",
                        dist.Multinomial(probs=word_probs),
                        obs=x)
                        # obs=x.reshape(-1, VOCAB_SIZE))

    # define the model p(x|z)p(z)
    def lda_model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z = pyro.sample("latent",
                            dist.Dirichlet(torch.from_numpy(self.alpha)))
            word_probs = self.decoder.forward(z)
            return pyro.sample("doc_words",
                        dist.Multinomial(probs=word_probs),
                        obs=x)


    def encoder_guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            # sample the latent code z
            pyro.sample("latent",
                        dist.Normal(z_loc, z_scale).to_event(1))
                        # dist.LogisticNormal(z_loc, z_scale).to_event(1))

    def mean_field_guide(self, x):
        with pyro.plate("data", x.shape[0]):
            # z_loc = pyro.param("z_loc", self.z_loc)
            # z_loc = pyro.param("z_loc", self.z_loc.repeat(x.shape[0], 1))
            z_loc = pyro.param("z_loc", x.new_zeros(torch.Size((x.shape[0], self.n_topics))))

            # z_scale = pyro.param("z_scale", self.z_scale)
            # z_scale = pyro.param("z_scale", self.z_scale.repeat(x.shape[0], 1))
            z_scale = pyro.param("z_scale", x.new_ones(torch.Size((x.shape[0], self.n_topics))))

            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def map_guide(self, x):
        with pyro.plate("data", x.shape[0]):
            z_loc = pyro.param("z_loc", self.z_loc)
            pyro.sample("latent", dist.Delta(z_loc).to_event(1))

    def reconstruct_with_vae_map(self, x, sample_z=False):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        if sample_z:
            # sample in latent space
            z_loc = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        word_probs = self.decoder(z_loc)
        return word_probs

    def load(self):
        state_dict = {}
        file_name = '_'.join([self.model_name, str(self.n_hidden_layers), str(self.n_hidden_units)]) + '.h5'
        h5f = h5py.File(os.path.join(self.results_dir, file_name), 'r')
        for i in range(self.n_hidden_layers):
            state_dict['encoder.enc_layers.{}.fc.weight'.format(i)] = torch.from_numpy(h5f['weights_{}'.format(i + 1)][()]).t()
            state_dict['encoder.enc_layers.{}.fc.bias'.format(i)] = torch.from_numpy(h5f['biases_{}'.format(i + 1)][()])

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

        state_dict['encoder.scale'] = torch.from_numpy(np.array(h5f['scale'][()]))

        if not self.topic_init:
            state_dict['decoder.topics'] = torch.from_numpy(h5f['topics'][()])

        h5f.close()

        return state_dict
