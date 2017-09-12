import tensorflow as tf
import time

import numpy as np
from PIL import Image

from munkres import Munkres, print_matrix

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/data/mnist', one_hot=True)

import os
os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '6'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

BATCH_SIZE = 100
LEARNING_RATE = 0.001
NUM_EPOCH = 1000
NUM_CLUSTER = 10
DIM_Z = 10
IMG_HEIGHT = 28
IMG_WIDTH = 28


class VAECluster:
    """Build the graph for VAECluster"""
    def __init__(self, batch_size, img_height, img_width,
                 hidden_dim, latent_dim, num_hidden, num_cluster, learning_rate):
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.input_dim = img_height * img_width
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_hidden = num_hidden
        self.num_cluster = num_cluster
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def __create_placeholders(self):
        """Create placeholders for input"""
        with tf.name_scope("input"):
            self.x = tf.placeholder(tf.float32, [self.batch_size, self.input_dim], name='x')
            self.y = tf.placeholder(tf.float32, [self.batch_size, self.num_cluster], name='y')
            self.x_reshape = tf.reshape(self.x, [1, self.batch_size, self.input_dim], 'x_reshape')
            self.x_duplicate = tf.tile(self.x_reshape, [self.num_cluster, 1, 1], 'x_duplicate')

    def __create_encoder_weight(self):
        """Create the weights for the encoder"""
        with tf.name_scope("encoder_weight"):
            self.encoder_w = []
            self.encoder_b = []
            self.previous_dim = self.input_dim
            for i_hidden in range(self.num_hidden):
                self.encoder_w.append(tf.get_variable('encoder_w' + str(i_hidden), [self.previous_dim, self.hidden_dim],
                                                      tf.float32, tf.contrib.layers.xavier_initializer(),
                                                      trainable=True))
                self.encoder_b.append(tf.Variable(tf.zeros([self.hidden_dim]), True, dtype=tf.float32,
                                                  name='encoder_b' + str(i_hidden)))
                self.previous_dim = self.hidden_dim
            self.mu_z_w = tf.get_variable('mu_z_w', [self.previous_dim, self.latent_dim], tf.float32,
                                          tf.contrib.layers.xavier_initializer(), trainable=True)
            self.mu_z_b = tf.Variable(tf.zeros([self.latent_dim]), True, name='mu_z_b')
            self.logsd_z_w = tf.get_variable('logsd_z_w', [self.previous_dim, self.latent_dim], tf.float32,
                                             tf.contrib.layers.xavier_initializer(), trainable=True)
            self.logsd_z_b = tf.Variable(tf.zeros([self.latent_dim]), True, name='logsd_z_b')

    def __create_decoder_weight(self):
        """Create the weights for the decoder"""
        with tf.name_scope("decoder_weight"):
            self.decoder_w = []
            self.decoder_b = []
            self.previous_dim = self.latent_dim
            self.decoder_w1 = []
            self.decoder_b1 = []
            for i_cluster in range(self.num_cluster):
                self.decoder_w1.append(tf.get_variable('decoder_w1_'+str(i_cluster), [self.previous_dim, self.hidden_dim],
                                                       tf.float32, tf.contrib.layers.xavier_initializer(),
                                                       trainable=True))
                self.decoder_b1.append(tf.Variable(tf.zeros([self.hidden_dim]), True, name='decoder_b1_'+str(i_cluster),
                                                   dtype=tf.float32))
            self.previous_dim = self.hidden_dim
            self.decoder_w.append(self.decoder_w1)
            self.decoder_b.append(self.decoder_b1)
            for i_hidden in range(1, self.num_hidden):
                self.decoder_w.append(tf.get_variable('decoder_w'+str(i_hidden), [self.previous_dim, self.hidden_dim],
                                                      tf.float32, tf.contrib.layers.xavier_initializer(),
                                                      trainable=True))
                self.decoder_b.append(tf.Variable(tf.zeros([self.hidden_dim]), True, name='decoder_b'+str(i_hidden),
                                                  dtype=tf.float32))
            self.decoder_x_w = tf.get_variable('decoder_x_w', [self.previous_dim, self.input_dim], tf.float32,
                                               tf.contrib.layers.xavier_initializer(), trainable=True)
            self.decoder_x_b = tf.Variable(tf.zeros([self.input_dim]), True, name='decoder_x_b', dtype=tf.float32)

    def __create_encoder(self):
        """Create the encoder"""
        with tf.name_scope("encoder"):
            self.encoder = []
            self.encoder_activate = []
            self.previous_dim = self.input_dim
            self.previous_tensor = self.x
            for i_hidden in range(self.num_hidden):
                self.encoder.append(tf.matmul(self.previous_tensor, self.encoder_w[i_hidden]) + self.encoder_b[i_hidden])
                self.encoder_activate.append(tf.nn.tanh(self.encoder[i_hidden]))
                self.previous_dim = self.hidden_dim
                self.previous_tensor = self.encoder_activate[i_hidden]
            self.mu_z = tf.matmul(self.previous_tensor, self.mu_z_w) + self.mu_z_b
            self.logsd_z = tf.matmul(self.previous_tensor, self.logsd_z_w) + self.logsd_z_b
            self.sd_z = tf.exp(self.logsd_z)

    def __create_decoder(self):
        """Create the decoder"""
        with tf.name_scope("sampling"):
            self.noise = tf.random_normal([self.batch_size, self.latent_dim], dtype=tf.float32, name='noise')
            self.z = tf.multiply(self.noise, self.sd_z) + self.mu_z
            self.previous_dim = self.latent_dim
            self.previous_tensor = self.z
        with tf.name_scope("decoder"):
            self.decoder1 = []
            for i_cluster in range(self.num_cluster):
                self.decoder1.append(tf.matmul(self.previous_tensor, self.decoder_w1[i_cluster])
                                     + self.decoder_b1[i_cluster])
            self.decoder1_concat = tf.concat(self.decoder1, 0, 'decoder1_concat')
            self.decoder1_activate = tf.nn.tanh(self.decoder1_concat)
            self.previous_dim = self.hidden_dim
            self.previous_tensor = self.decoder1_activate
            self.decoder = [self.decoder1_concat]
            self.decoder_activate = [self.decoder1_activate]
            for i_hidden in range(1, self.num_hidden):
                self.decoder.append(tf.matmul(self.previous_tensor, self.decoder_w[i_hidden]) + self.decoder_b[i_hidden])
                self.decoder_activate.append(tf.nn.tanh(self.decoder[i_hidden]))
                self.previous_dim = self.hidden_dim
                self.previous_tensor = self.decoder_activate[i_hidden]
            self.decoder_final = tf.matmul(self.previous_tensor, self.decoder_x_w) + self.decoder_x_b
            self.x_hat = tf.nn.sigmoid(self.decoder_final)
            self.x_hat_reshape = tf.reshape(self.x_hat, [self.num_cluster, self.batch_size, self.input_dim])

    def __create_generator(self):
        """Create generator"""
        with tf.name_scope("generator"):
            self.generator_decoder1 = []
            self.previous_tensor = self.noise
            self.previous_dim = self.latent_dim
            for i_cluster in range(self.num_cluster):
                self.generator_decoder1.append(tf.matmul(self.previous_tensor, self.decoder_w1[i_cluster])
                                               + self.decoder_b1[i_cluster])
            self.generator_decoder1_concat = tf.concat(self.generator_decoder1, 0, 'generator_decoder1_concat')
            self.generator_decoder1_activate = tf.nn.tanh(self.generator_decoder1_concat)
            self.previous_tensor = self.generator_decoder1_activate
            self.previous_dim = self.hidden_dim
            self.generator_decoder = [self.generator_decoder1_concat]
            self.generator_decoder_activate = [self.generator_decoder1_activate]
            for i_hidden in range(1, self.num_hidden):
                self.generator_decoder.append(tf.matmul(self.previous_tensor, self.decoder_w[i_hidden])
                                              + self.decoder_b[i_hidden])
                self.generator_decoder_activate.append(tf.nn.tanh(self.generator_decoder[i_hidden]))
                self.previous_dim = self.hidden_dim
                self.previous_tensor = self.generator_decoder_activate[i_hidden]
            self.generator_decoder_final = tf.matmul(self.previous_tensor, self.decoder_x_w) + self.decoder_x_b
            self.generator_x_hat = tf.nn.sigmoid(self.generator_decoder_final)
            self.generator_x_hat_reshape = tf.reshape(self.generator_x_hat, [self.num_cluster*self.batch_size,
                                                                             self.img_height, self.img_width, 1])

    def __create_posterior(self):
        """Create posterior"""
        with tf.name_scope("prior"):
            self.prior_logit = tf.Variable(tf.zeros([self.num_cluster, 1]), False, dtype=tf.float32, name='prior_logit')
            self.prior = tf.nn.softmax(self.prior_logit, 0, 'prior')
        with tf.name_scope("posterior"):
            self.logp_x_yz_dim = - tf.multiply(self.x_duplicate, tf.log(self.x_hat_reshape + 1e-12)) \
                                - tf.multiply(1-self.x_duplicate, tf.log(1-self.x_hat_reshape + 1e-12))
            self.logp_x_yz = tf.reduce_sum(self.logp_x_yz_dim, 2)
            self.logp_xy_z = self.logp_x_yz + tf.log(self.prior)
            self.posterior = tf.nn.softmax(self.logp_xy_z, 0, 'posterior')
            self.label = tf.argmax(self.posterior, 0, name='label')
            self.label_gt = tf.argmax(self.y, 1, name='label_gt')

    def __create_loss(self):
        """Create loss"""
        with tf.name_scope("loss"):
            self.klloss_mu = tf.square(self.mu_z) / 2
            self.klloss_sd = tf.square(self.sd_z) / 2 - self.logsd_z - 0.5
            self.klloss = tf.reduce_sum(self.klloss_mu + self.klloss_sd, name='klloss')/self.batch_size
            self.max_logp_xy_z = tf.reduce_max(self.logp_xy_z, 0, True)
            self.res_logp_xy_z = self.logp_xy_z - tf.tile(self.max_logp_xy_z, [self.num_cluster, 1])
            self.generate_loss = tf.reduce_sum(self.max_logp_xy_z
                                               + tf.log(tf.reduce_sum(tf.exp(self.res_logp_xy_z), 0, True))) / self.batch_size
            self.loss = self.klloss + self.generate_loss

    def __create_optimizer(self):
        """Create optimizer"""
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def __create_summary(self):
        """Create summary"""
        with tf.name_scope("summary"):
            self.summary_loss = tf.summary.scalar('loss', self.loss)
            self.summary_generate_loss = tf.summary.scalar('generate_loss', self.generate_loss)
            self.summary_klloss = tf.summary.scalar('klloss', self.klloss)
            self.summary = tf.summary.merge([self.summary_loss, self.summary_generate_loss, self.summary_klloss])
        with tf.name_scope("summary_generator"):
            self.summary_generator = tf.summary.image('generate_sample', self.generator_x_hat_reshape, 30)

    def build_graph(self):
        self.__create_placeholders()
        self.__create_encoder_weight()
        self.__create_decoder_weight()
        self.__create_encoder()
        self.__create_decoder()
        self.__create_generator()
        self.__create_posterior()
        self.__create_loss()
        self.__create_optimizer()
        self.__create_summary()


def train_model(model, data, epoch):
    saver = tf.train.Saver()
    if not os.path.exists('./model'):
        os.mkdir('./model')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('./model/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        writer = tf.summary.FileWriter('./graph', sess.graph)
        initial_step = model.global_step.eval()
        batch_per_epoch = int(data.num_examples / model.batch_size)
        num_steps = batch_per_epoch * epoch
        total_loss = 0
        for index in range(initial_step, initial_step+num_steps):
            x_batch, y_batch = data.next_batch(model.batch_size)
            feed_dict = {model.x: x_batch}
            batch_loss, _, summary = sess.run([model.loss, model.optimizer, model.summary], feed_dict=feed_dict)
            total_loss += batch_loss
            writer.add_summary(summary, index)
            if (index+1)%batch_per_epoch == 0:
                print('Global step = {0}. Average loss = {1}.'.format(index, total_loss/batch_per_epoch))
                total_loss = 0
                saver.save(sess, 'model/VAE_Clustering' + str(index))
                summary = sess.run(model.summary_generator)
                writer.add_summary(summary, index)


def test_model(model, data):
    y_pred = list()
    y_gt = list()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('./model/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        batch_per_epoch = int(data.num_examples / model.batch_size)
        for index in range(batch_per_epoch):
            x_batch, y_batch = data.next_batch(model.batch_size)
            feed_dict = {model.x: x_batch, model.y: y_batch}
            prior, posterior, label, label_gt = sess.run([model.prior, model.posterior, model.label, model.label_gt], feed_dict=feed_dict)
            y_gt += list(label_gt)
            y_pred += list(label)
    assignment = np.zeros([model.num_cluster, model.num_cluster])
    for i in range(len(y_pred)):
        assignment[y_pred[i]][y_gt[i]] += 1
    m = Munkres()
    indexes = m.compute(-assignment)
    total_cost = 0
    for row,col in indexes:
        total_cost += assignment[row][col]
    accuracy = total_cost / len(y_gt)
    print('{0} correct out of {1} samples. Accuracy = {2}.'.format(total_cost, len(y_gt), accuracy))


def generate_sample(model):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('./model/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        generate_x = sess.run(model.generator_x_hat_reshape)
        img = np.ones([300, 300], np.float32)
        for i in range(10):
            for j in range(10):
                img[i*30+1:i*30+29, j*30+1:j*30+29] = generate_x[i+j*BATCH_SIZE, :, :, 0]
        rescaled = (255 * img).astype(np.uint8)
        im = Image.fromarray(rescaled)
        img_name = 'generate/sample.png'
        im.save(img_name)


def main(dataset):
    model = VAECluster(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH,
                       100, 10, 2, NUM_CLUSTER, LEARNING_RATE)
    model.build_graph()
    train_model(model, dataset.train, NUM_EPOCH)
    test_model(model, dataset.train)
    test_model(model, dataset.test)
    generate_sample(model)


if __name__ == '__main__':
    main(mnist)
