import tensorflow as tf
from tensorflow.contrib import layers
import os
import data_util as du
import math
import numpy as np
from munkres import Munkres, print_matrix

HALF_LOG_TWO_PI = math.log(2 * math.pi) / 2


class VAEClustering:
    def __init__(self, batch_size, dimension, cluster_num, data_type):
        self.batch_size = batch_size
        self.dimension = dimension
        assert len(dimension) >= 2
        self.input_dim = dimension[0]
        self.hidden_dim = dimension[1:-1]
        self.latent_dim = dimension[-1]
        self.data_type = data_type
        self.cluster_num = cluster_num
        self.protector = 1e-6
        self.__build_network()

    def __build_network(self):
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [self.batch_size, self.input_dim], 'x')
        previous_dim = self.input_dim
        previous_tensor = self.x
        with tf.name_scope('encoder'):
            for i in range(len(self.hidden_dim)):
                with tf.variable_scope('encoder'):
                    w = tf.get_variable('w' + str(i), [previous_dim, self.hidden_dim[i]], tf.float32,
                                        layers.xavier_initializer(), trainable=True)
                    b = tf.get_variable('b' + str(i), [self.hidden_dim[i]], tf.float32,
                                        tf.zeros_initializer(), trainable=True)
                    with tf.name_scope('hidden' + str(i)):
                        h = tf.matmul(previous_tensor, w) + b
                        previous_tensor = tf.nn.relu(h)
                        previous_dim = self.hidden_dim[i]
        with tf.name_scope('latent'):
            with tf.variable_scope('latent'):
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
                        self.sd_z = tf.nn.sigmoid(self.logsd_z)#tf.exp(self.logsd_z)
        with tf.name_scope('sample'):
            self.noise = tf.random_normal([self.batch_size, self.latent_dim], dtype=tf.float32)
            self.z = self.mu_z + self.noise * self.sd_z
        previous_dim = self.latent_dim
        previous_tensor = self.z
        with tf.name_scope('decoder'):
            for i in range(len(self.hidden_dim)):
                with tf.variable_scope('decoder'):
                    w = tf.get_variable('w' + str(i), [previous_dim, self.hidden_dim[-i-1]], tf.float32,
                                        layers.xavier_initializer(), trainable=True)
                    b = tf.get_variable('b' + str(i), [self.hidden_dim[-i-1]], tf.float32,
                                        tf.zeros_initializer(), trainable=True)
                    with tf.name_scope('hidden' + str(i)):
                        h = tf.matmul(previous_tensor, w) + b
                        previous_tensor = tf.nn.relu(h)
                        previous_dim = self.hidden_dim[-i-1]
        with tf.name_scope('x_hat'):
            with tf.variable_scope('x_hat'):
                w = tf.get_variable('w', [previous_dim, self.input_dim], tf.float32,
                                    layers.xavier_initializer(), trainable=True)
                b = tf.get_variable('b', [self.input_dim], tf.float32,
                                    tf.zeros_initializer(), trainable=True)
                if self.data_type == 'sigmoid':
                    self.x_hat_logit = tf.matmul(previous_tensor, w) + b
                    self.x_hat = tf.nn.sigmoid(self.x_hat_logit)
                else:
                    self.x_hat = tf.matmul(previous_tensor, w) + b
        with tf.variable_scope('centroids'):
            with tf.name_scope('centroids'):
                self.prior_logit = tf.get_variable('prior_logit', [self.cluster_num], tf.float32,
                                                   tf.zeros_initializer(), trainable=False)
                self.prior = tf.nn.softmax(self.prior_logit)
                self.centroids = tf.get_variable('centroids', [self.cluster_num, self.latent_dim], tf.float32,
                                                 tf.random_normal_initializer(), trainable=False)
        with tf.name_scope('loss'):
            with tf.name_scope('generate_loss'):
                if self.data_type == 'sigmoid':
                    self.generate_loss \
                        = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x,
                                                                                logits=self.x_hat_logit)) \
                          / self.batch_size
                else:
                    self.generate_loss = tf.reduce_sum(tf.square(self.x_hat - self.x)) / self.batch_size / 2
            with tf.name_scope('kl_loss'):
                with tf.name_scope('logq'):
                    self.logq = tf.reduce_sum(-HALF_LOG_TWO_PI - tf.log(self.sd_z + self.protector)
                                              - tf.square(self.noise)/2)
                with tf.name_scope('logp'):
                    tile_z = tf.tile(tf.reshape(self.z, [self.batch_size, 1, self.latent_dim]),
                                     [1, self.cluster_num, 1])
                    tile_centroid = tf.tile(tf.reshape(self.centroids, [1, self.cluster_num, self.latent_dim]),
                                            [self.batch_size, 1, 1])
                    tile_prior = tf.tile(tf.reshape(self.prior, [1, self.cluster_num]), [self.batch_size, 1])
                    p_z_given_y = 1 / math.pi / (1 + tf.reduce_sum(tf.square(tile_z - tile_centroid), 2))
                    self.inspect = p_z_given_y
                    self.p_zy = p_z_given_y * tile_prior
                    self.p_z = tf.reduce_sum(self.p_zy, 1)
                    self.logp = tf.reduce_sum(tf.log(self.p_z + self.protector))
                self.kl_loss = (self.logq - self.logp) / self.batch_size
            self.loss = self.generate_loss + self.kl_loss
        with tf.name_scope('posterior'):
            self.posterior = self.p_zy / tf.tile(tf.reshape(self.p_z, [self.batch_size, 1]), [1, self.cluster_num])
            self.label_hat = tf.arg_max(self.posterior, -1)
        with tf.name_scope('summary'):
            kl_summary = tf.summary.scalar('kl', self.kl_loss)
            generate_summary = tf.summary.scalar('generate', self.generate_loss)
            loss_summary = tf.summary.scalar('loss', self.loss)
            label_summary = tf.summary.histogram('label', self.label_hat)
            sd_summary = tf.summary.histogram('sd_z', self.sd_z)
            self.summary = tf.summary.merge([kl_summary, generate_summary, loss_summary, label_summary, sd_summary])


class VAEOptimizer:
    def __init__(self, model):
        with tf.variable_scope('global_step'):
            self.global_step = tf.Variable(0, False, name='gloabl_step', dtype=tf.float32)
        with tf.name_scope('learning_rate'):
            self.learning_rate = tf.placeholder(tf.float32, [], 'learning_rate')
        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=0.0001).minimize(model.loss)


def train_model(model, x, num_epoch):
    optimizer = VAEOptimizer(model)
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
        iteration = 0
        for epoch in range(num_epoch):
            lr = 0.0002 * math.pow(0.9, math.floor(epoch/10))
            x = du.shuffle_data(x)
            total_loss = 0
            for i in range(iteration_per_epoch):
                feed_dict = {model.x: x[i*model.batch_size:(i+1)*model.batch_size, :],
                             optimizer.learning_rate: lr}
                sd_z, loss, _, summary = sess.run([model.sd_z, model.loss, optimizer.optimizer, model.summary],
                                            feed_dict=feed_dict)
                writer.add_summary(summary, iteration)
                iteration += 1
                total_loss += loss
            total_loss /= iteration_per_epoch
            print('epoch = {0}, loss = {1}.'.format(epoch, total_loss))

            if epoch % 10 == 0:
                saver.save(sess, './model/VAE' + str(epoch), optimizer.global_step)


def test_model(model, x, y):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('model/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        iteration_per_epoch = int(math.floor(x.shape[0] / model.batch_size))
        label = list()
        for i in range(iteration_per_epoch):
            feed_dict = {model.x: x[i*model.batch_size:(i+1)*model.batch_size, :]}
            label_batch = sess.run(model.label_hat, feed_dict=feed_dict)
            label += list(label_batch)

        assignment = np.zeros([model.cluster_num, model.cluster_num])
        label_gt = np.argmax(y, 1)
        for i in range(len(label)):
            idx = int(label_gt[i])
            assignment[label[i]][idx] += 1
        m = Munkres()
        indexes = m.compute(-assignment)
        total_cost = 0
        for row, col in indexes:
            total_cost += assignment[row][col]
        accuracy = total_cost / len(label)
        print('{0} correct out of {1} samples. Accuracy = {2}.'.format(total_cost, len(label), accuracy))


def main():
    model_dimension = [784, 500, 500, 2000, 10]
    model = VAEClustering(100, model_dimension, 10, 'sigmoid')

    train_data, train_label = du.load_mnist_data('training')
    test_data, test_label = du.load_mnist_data('testing')

#    train_model(model, train_data, 100)
    test_model(model, train_data, train_label)
    test_model(model, test_data, test_label)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    main()
