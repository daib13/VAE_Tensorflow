import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from munkres import Munkres, print_matrix
import math
from tensorflow.contrib import layers

import sys
sys.path.insert(0, '../util')
import data_util as du

import os


class VAEClustering:
    def __init__(self, batch_size, dimension, cluster_num, alpha=0.01, data_type='sigmoid'):
        self.batch_size = batch_size
        self.dimension = dimension
        assert len(dimension) >= 2
        self.input_dim = dimension[0]
        self.hidden_dim = dimension[1:-1]
        self.latent_dim = dimension[-1]
        self.cluster_num = cluster_num
        assert self.latent_dim % cluster_num == 0
        self.latent_dim_per_cluster = self.latent_dim / cluster_num
        self.alpha = alpha
        self.data_type = data_type
        self.protector = 1e-20

    def __create_placeholder(self):
        with tf.name_scope('x'):
            self.x = tf.placeholder(tf.float32, [self.batch_size, self.input_dim], 'x')

    def __create_encoder(self):
        previous_dim = self.input_dim
        previous_tensor = self.x
        with tf.variable_scope('encoder'):
            with tf.name_scope('encoder'):
                for i in range(len(self.hidden_dim)):
                    with tf.variable_scope('encoder' + str(i)):
                        w = tf.get_variable('w', [previous_dim, self.hidden_dim[i]], tf.float32,
                                            layers.xavier_initializer(), trainable=True)
                        b = tf.get_variable('b', [self.hidden_dim[i]], tf.float32,
                                            tf.zeros_initializer(), trainable=True)
                        with tf.name_scope('encoder' + str(i)):
                            h = tf.matmul(previous_tensor, w) + b
                            previous_tensor = tf.nn.relu(h)
                            previous_dim = self.hidden_dim[i]
        with tf.variable_scope('mu_z'):
            w = tf.get_variable('w', [previous_dim, self.latent_dim], tf.float32,
                                layers.xavier_initializer(), trainable=True)
            b = tf.get_variable('b', [self.latent_dim], tf.float32,
                                tf.zeros_initializer(), trainable=True)
            with tf.name_scope('mu_z'):
                self.mu_z = tf.matmul(previous_tensor, w) + b
        with tf.variable_scope('sd_z'):
            w = tf.get_variable('w', [previous_dim, self.latent_dim], tf.float32,
                                layers.xavier_initializer(), trainable=True)
            b = tf.get_variable('b', [self.latent_dim], tf.float32,
                                tf.zeros_initializer(), trainable=True)
            with tf.name_scope('sd_z'):
                self.logsd_z = tf.matmul(previous_tensor, w) + b
                self.sd_z = tf.exp(self.logsd_z)

    def __create_decoder(self):
        with tf.name_scope('sample'):
            self.noise = tf.random_normal([self.batch_size, self.latent_dim], dtype=tf.float32, name='noise')
            self.z = self.noise * self.sd_z + self.mu_z
        previous_dim = self.latent_dim
        previous_tensor = self.z
        with tf.variable_scope('decoder'):
            with tf.name_scope('decoder'):
                for i in range(len(self.hidden_dim)):
                    with tf.variable_scope('decoder' + str(i)):
                        w = tf.get_variable('w', [previous_dim, self.hidden_dim[-i-1]], tf.float32,
                                            layers.xavier_initializer(), trainable=True)
                        b = tf.get_variable('b', [self.hidden_dim[-i-1]], tf.float32,
                                            tf.zeros_initializer(), trainable=True)
                        with tf.name_scope('decoder' + str(i)):
                            h = tf.matmul(previous_tensor, w) + b
                            previous_tensor = tf.nn.relu(h)
                            previous_dim = self.hidden_dim[-i-1]
        with tf.variable_scope('x_hat'):
            w = tf.get_variable('w', [previous_dim, self.input_dim], tf.float32,
                                layers.xavier_initializer(), trainable=True)
            b = tf.get_variable('b', [self.input_dim], tf.float32,
                                tf.zeros_initializer(), trainable=True)
            with tf.name_scope('x_hat'):
                if self.data_type == 'sigmoid':
                    self.x_hat_logit = tf.matmul(previous_tensor, w) + b
                    self.x_hat = tf.nn.sigmoid(self.x_hat_logit)
                else:
                    self.x_hat_logit = None
                    self.x_hat = tf.matmul(previous_tensor, w) + b

    def __create_loss(self):
        with tf.variable_scope('prior'):
            self.prior_logit = tf.get_variable('prior_logit', [self.cluster_num], tf.float32,
                                               tf.zeros_initializer(), trainable=True)
            sd = np.ones([self.cluster_num, self.latent_dim]) * self.alpha
            for i in range(self.cluster_num):
                sd[i, int(self.latent_dim_per_cluster*i):int(self.latent_dim_per_cluster*(i+1))] = 1
            self.sd = tf.constant(sd, tf.float32)
            with tf.name_scope('prior'):
                self.prior = tf.nn.softmax(self.prior_logit)

        with tf.name_scope('generate_loss'):
            if self.data_type == 'sigmoid':
                self.generate_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x,
                                                                                           logits=self.x_hat_logit)) \
                                     / self.batch_size
            else:
                self.generate_loss = tf.reduce_sum(tf.square(self.x, self.x_hat) / 2) / self.batch_size

        with tf.name_scope('kl_loss'):
            self.logq = tf.reduce_sum(-self.logsd_z - tf.square(self.noise) / 2, -1)

            tile_prior_logit = tf.tile(tf.reshape(tf.log(self.prior + self.protector), [1, self.cluster_num]),
                                       [self.batch_size, 1])
            tile_sd = tf.tile(tf.reshape(self.sd, [1, self.cluster_num, self.latent_dim]),
                              [self.batch_size, 1, 1])
            tile_z = tf.tile(tf.reshape(self.z, [self.batch_size, 1, self.latent_dim]), [1, self.cluster_num, 1])
            self.pz_given_y_logit = tf.reduce_sum(-tf.log(tile_sd) - tf.square(tile_z / tile_sd) / 2, -1)
            self.logp = self.pz_given_y_logit + tile_prior_logit
            self.posterior = tf.nn.softmax(self.logp)
            self.y_pred = tf.argmax(self.posterior, -1)
            self.logp_max = tf.reduce_max(self.logp, -1)
            self.logp_res = tf.reduce_sum(tf.exp(self.logp - tf.tile(tf.reshape(self.logp_max, [self.batch_size, 1]),
                                                                     [1, self.cluster_num])), -1)
            self.logp_exploitation = self.logp_max + tf.log(self.logp_res)
            self.logp_exploration = tf.reduce_sum(self.logp, -1) / self.cluster_num \
                                    + tf.log(tf.cast(self.cluster_num, tf.float32))
            self.exploration_ratio = tf.placeholder(tf.float32, [], 'exploration')
            self.logp_mix = (1 - self.exploration_ratio) * self.logp_exploitation \
                            + self.exploration_ratio * self.logp_exploration

            self.kl_loss = tf.reduce_sum(self.logq - self.logp_mix) / self.batch_size

        with tf.name_scope('posterior_loss'):
            tile_prior = tf.tile(tf.reshape(self.prior + self.protector, [1, self.cluster_num]), [self.batch_size, 1])
            posterior_square = tf.square(self.posterior)
            unnorm_posterior_gt = posterior_square / tile_prior
            norm_posterior_gt = unnorm_posterior_gt / tf.tile(tf.reduce_sum(unnorm_posterior_gt, -1, True),
                                                              [1, self.cluster_num])
            self.posterior_gt = tf.stop_gradient(norm_posterior_gt)
            self.posterior_loss = -tf.reduce_sum(self.posterior_gt * tf.log(self.posterior + self.protector)) \
                                  / self.batch_size

        with tf.name_scope('loss'):
            self.loss = self.generate_loss + self.kl_loss + self.posterior_loss

    def __create_summary(self):
        with tf.name_scope('summary'):
            self.exploration_loss = tf.summary.scalar('summary_exploration',
                                                      tf.reduce_sum(self.logp_exploration) / self.batch_size)
            self.exploitation_loss = tf.summary.scalar('summary_exploitation',
                                                       tf.reduce_sum(self.logp_exploitation) / self.batch_size)
            self.summary_kl_loss = tf.summary.scalar('summary_kl_loss', self.kl_loss)
            self.summary_generate_loss = tf.summary.scalar('summary_generate_loss', self.generate_loss)
            self.posterior_loss = tf.summary.scalar('summary_posterior_loss', self.posterior_loss)
            self.summary_loss = tf.summary.scalar('summary_loss', self.loss)
            self.summary_label = tf.summary.histogram('summary_label', self.y_pred)
            self.summary = tf.summary.merge([self.summary_kl_loss, self.summary_generate_loss, self.summary_loss,
                                             self.exploitation_loss, self.exploration_loss, self.summary_label,
                                             self.posterior_loss],
                                            name='summary')

    def build_network(self):
        self.__create_placeholder()
        self.__create_encoder()
        self.__create_decoder()
        self.__create_loss()
        self.__create_summary()


class VAEOptimizer:
    def __init__(self, model):
        with tf.name_scope('optimizer'):
            with tf.variable_scope('optimizer'):
                self.global_step = tf.Variable(0, False, name='global_step')
                self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate,
                                                        epsilon=0.0001).minimize(model.loss, self.global_step)


def test_model(model, x, y):
    y_pred = list()
    y_gt = np.argmax(y, 1)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('./model/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        iteration_per_epoch = int(math.floor(x.shape[0] / model.batch_size))
        for i in range(iteration_per_epoch):
            feed_dict = {model.x: x[i*model.batch_size:(i+1)*model.batch_size, :]}
            label, posterior = sess.run([model.y_pred, model.posterior], feed_dict=feed_dict)
            y_pred += list(label)

    assignment = np.zeros([model.cluster_num, model.cluster_num])
    for i in range(len(y_pred)):
        assignment[y_pred[i]][y_gt[i]] += 1
    m = Munkres()
    indexes = m.compute(-assignment)
    total_cost = 0
    for row, col in indexes:
        total_cost += assignment[row][col]
    accuracy = total_cost / len(y_gt)
    print('{0} correct out of {1} samples. Accuracy = {2}.'.format(total_cost, len(y_gt), accuracy))

    new_label = y_pred.copy()
    for row, col in indexes:
        idxes = [i for i, x in enumerate(y_pred) if x == row]
        for idx in idxes:
            new_label[idx] = col
    correct = [l1 == l2 for l1, l2 in zip(new_label, y_gt)]
    return correct


def train_model(model, optimizer, x, y, num_epoch, exploration_ratio):
    saver = tf.train.Saver()
    if not os.path.exists('./model'):
        os.mkdir('./model')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('model/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        writer = tf.summary.FileWriter('graph', sess.graph)

        iteration_per_epoch = int(math.floor(x.shape[0] / model.batch_size))
        for epoch in range(num_epoch):
            total_loss = 0
            x, y = du.shuffle_data(x, y)
            for i in range(iteration_per_epoch):
                feed_dict = {model.x: x[i*model.batch_size:(i+1)*model.batch_size, :],
                             optimizer.learning_rate: 0.0003,
                             model.exploration_ratio: exploration_ratio}
                batch_loss, _, summary = sess.run([model.loss, optimizer.optimizer, model.summary], feed_dict=feed_dict)
                writer.add_summary(summary, optimizer.global_step.eval(session=sess))
                total_loss += batch_loss
            print('Epoch = {0}, loss = {1}.'.format(epoch, total_loss / iteration_per_epoch))
            if (epoch + 1) % 10 == 0:
                saver.save(sess, 'model/VAE' + str(epoch + 1))
                test_model(model, x, y)


def extract_prior(model):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('model/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        prior = sess.run(model.prior)
        print(prior)


def extract_posterior(model, x):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('model/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

            posterior = np.zeros(model.cluster_num)
            iteration_per_epoch = int(math.floor(x.shape[0] / model.batch_size))
            for i in range(iteration_per_epoch):
                feed_dict = {model.x: x[i*model.batch_size:(i+1)*model.batch_size, :]}
                batch_posterior = sess.run(model.posterior, feed_dict=feed_dict)
                posterior += np.sum(batch_posterior, 0) / model.batch_size
            posterior /= iteration_per_epoch
            print(posterior)


def extract_max_posterior(model, x, y):
    correct = test_model(model, x, y)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('model/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

            iteration_per_epoch = int(math.floor(x.shape[0] / model.batch_size))
            iter = 0
            for i in range(iteration_per_epoch):
                feed_dict = {model.x: x[i*model.batch_size:(i+1)*model.batch_size, :]}
                batch_posterior = sess.run(model.posterior, feed_dict=feed_dict)
                max_posterior = np.max(batch_posterior, 1)
                for j in range(max_posterior.size):
                    print('{0}\t{1}'.format(max_posterior[j], correct[iter]))
                    iter += 1


def main(exploration_ratio):
    model = VAEClustering(100, [784, 500, 500, 2000, 10], 10, 0.01, 'sigmoid')
    model.build_network()

    optimizer = VAEOptimizer(model)

    x_train, y_train = du.load_mnist_data('training')
    x_test, y_test = du.load_mnist_data('testing')
    train_model(model, optimizer, x_train, y_train, 1000, exploration_ratio)
    test_model(model, x_train, y_train)
    test_model(model, x_test, y_test)

#    extract_prior(model)
#    extract_posterior(model, x_train)
#    extract_max_posterior(model, x_train, y_train)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main(0.1)
#    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]
#    origin_stdout = sys.stdout
#    fid = open('max_posterior.txt', 'w')
#    sys.stdout = fid

#    main(eval(sys.argv[1]))
#    sys.stdout = origin_stdout
#    fid.close()
