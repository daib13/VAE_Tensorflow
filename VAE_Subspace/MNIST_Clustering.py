import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from munkres import Munkres, print_matrix

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/data/mnist', one_hot=True)

import os
os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '6'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

BATCH_SIZE = 100
INPUT_DIM = 784
LATENT_DIM_PER_CLUSTER = 2
CLUSTER_NUM = 10
LEARNING_RATE = 0.001
NUM_EPOCH = 100


class VAEClustering:
    def __init__(self, batch_size, input_dim, latent_dim_per_cluster, cluster_num, learning_rate):
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.latent_dim_per_cluster = latent_dim_per_cluster
        self.cluster_num = cluster_num
        self.latent_dim = self.latent_dim_per_cluster * self.cluster_num
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def __create_placeholder(self):
        with tf.name_scope("data"):
            self.x = tf.placeholder(tf.float32, [self.batch_size, self.input_dim], 'x')
        with tf.name_scope("label"):
            self.y = tf.placeholder(tf.int32, [self.batch_size, self.cluster_num], 'y')
            self.label_gt = tf.argmax(self.y, 1, name='label_gt')

    def __create_encoder(self):
        with tf.name_scope("encoder1"):
            self.encoder1_w = tf.get_variable('encoder1_w', [self.input_dim, 500], tf.float32,
                                              tf.contrib.layers.xavier_initializer(), trainable=True)
            self.encoder1_b = tf.Variable(tf.zeros([500]), True, dtype=tf.float32, name='encoder1_b')
            self.encoder1 = tf.matmul(self.x, self.encoder1_w) + self.encoder1_b
            self.encoder1_relu = tf.nn.relu(self.encoder1)
        with tf.name_scope("encoder2"):
            self.encoder2_w = tf.get_variable('encoder2_w', [500, 500], tf.float32,
                                              tf.contrib.layers.xavier_initializer(), trainable=True)
            self.encoder2_b = tf.Variable(tf.zeros([500]), True, dtype=tf.float32, name='encoder2_b')
            self.encoder2 = tf.matmul(self.encoder1_relu, self.encoder2_w) + self.encoder2_b
            self.encoder2_relu = tf.nn.relu(self.encoder2)
        with tf.name_scope("encoder3"):
            self.encoder3_w = tf.get_variable("encoder3_w", [500, 2000], tf.float32,
                                              tf.contrib.layers.xavier_initializer(), trainable=True)
            self.encoder3_b = tf.Variable(tf.zeros([2000]), True, dtype=tf.float32, name='encoder3_b')
            self.encoder3 = tf.matmul(self.encoder2_relu, self.encoder3_w) + self.encoder3_b
            self.encoder3_relu = tf.nn.relu(self.encoder3)

    def __create_latent(self):
        with tf.name_scope("mu_z"):
            self.mu_z_w = tf.get_variable("mu_z_w", [2000, self.latent_dim], tf.float32,
                                          tf.contrib.layers.xavier_initializer(), trainable=True)
            self.mu_z_b = tf.Variable(tf.zeros([self.latent_dim]), True, dtype=tf.float32, name='mu_z_b')
            self.mu_z = tf.matmul(self.encoder3_relu, self.mu_z_w) + self.mu_z_b
        with tf.name_scope("sd_z"):
            self.logsd_z_w = tf.get_variable("logsd_z_w", [2000, self.latent_dim], tf.float32,
                                             tf.contrib.layers.xavier_initializer(), trainable=True)
            self.logsd_z_b = tf.Variable(tf.zeros([self.latent_dim]), True, dtype=tf.float32, name='logsd_z_b')
            self.logsd_z = tf.matmul(self.encoder3_relu, self.logsd_z_w) + self.logsd_z_b
            self.sd_z = tf.exp(self.logsd_z)
        with tf.name_scope("sd_gt"):
            self.mu_z_dis = tf.square(self.mu_z)
#            self.sd_z_dis = self.sd_z - self.logsd_z - 1
            self.mutual_info_dim = self.mu_z_dis# + self.sd_z_dis
            self.mutual_info_dim_reshape = tf.reshape(self.mutual_info_dim,
                                                      [self.batch_size, self.cluster_num, self.latent_dim_per_cluster])
            self.mutual_info_cluster = tf.exp(tf.reduce_sum(self.mutual_info_dim_reshape, 2))
            self.mutual_info_sum = tf.reduce_sum(self.mutual_info_cluster, 1, keep_dims=True)
            self.mutual_info_sum_tile = tf.tile(self.mutual_info_sum, [1, self.cluster_num])
            self.posterior = tf.divide(self.mutual_info_cluster, self.mutual_info_sum_tile, 'posterior')
            self.posterior_reshape = tf.reshape(self.posterior, [self.batch_size, self.cluster_num, 1])
            self.sd_gt_tile = tf.tile(self.posterior_reshape, [1, 1, self.latent_dim_per_cluster])
            self.sd_gt = tf.add(0.01, tf.reshape(self.sd_gt_tile, [self.batch_size, self.latent_dim]), 'sd_gt')
        with tf.name_scope("label"):
            self.label = tf.argmax(self.posterior, 1, name='label')

    def __create_decoder(self):
        with tf.name_scope("sample"):
            self.noise = tf.random_normal([self.batch_size, self.latent_dim], dtype=tf.float32, name='noise')
            self.z = tf.multiply(self.noise, self.sd_z) + self.mu_z
        with tf.name_scope("decoder1"):
            self.decoder1_w = tf.get_variable("decoder1_w", [self.latent_dim, 2000], tf.float32,
                                              tf.contrib.layers.xavier_initializer(), trainable=True)
            self.decoder1_b = tf.Variable(tf.zeros([2000]), True, dtype=tf.float32, name='decoder1_b')
            self.decoder1 = tf.matmul(self.z, self.decoder1_w) + self.decoder1_b
            self.decoder1_relu = tf.nn.relu(self.decoder1)
        with tf.name_scope("decoder2"):
            self.decoder2_w = tf.get_variable("decoder2_w", [2000, 500], tf.float32,
                                              tf.contrib.layers.xavier_initializer(), trainable=True)
            self.decoder2_b = tf.Variable(tf.zeros([500]), True, dtype=tf.float32, name='decoder2_b')
            self.decoder2 = tf.matmul(self.decoder1_relu, self.decoder2_w) + self.decoder2_b
            self.decoder2_relu = tf.nn.relu(self.decoder2)
        with tf.name_scope("decoder3"):
            self.decoder3_w = tf.get_variable("decoder3_w", [500, 500], tf.float32,
                                              tf.contrib.layers.xavier_initializer(), trainable=True)
            self.decoder3_b = tf.Variable(tf.zeros([500]), True, dtype=tf.float32, name='decoder3_b')
            self.decoder3 = tf.matmul(self.decoder2_relu, self.decoder3_w) + self.decoder3_b
            self.decoder3_relu = tf.nn.relu(self.decoder3)
        with tf.name_scope("decoder_final"):
            self.decoder_final_w = tf.get_variable("decoder_final_w", [500, 784], tf.float32,
                                                   tf.contrib.layers.xavier_initializer(), trainable=True)
            self.decoder_final_b = tf.Variable(tf.zeros([784]), True, dtype=tf.float32, name='decoder_final_b')
            self.decoder_final = tf.matmul(self.decoder3_relu, self.decoder_final_w) + self.decoder_final_b
            self.x_hat = tf.nn.sigmoid(self.decoder_final)

    def __create_loss(self):
        with tf.name_scope("loss"):
            self.kl_loss = tf.reduce_sum(tf.square(tf.divide(self.mu_z, self.sd_gt))
                                         + tf.square(tf.divide(self.sd_z, self.sd_gt))
                                         - self.logsd_z + tf.log(self.sd_gt))
            self.generate_loss = -tf.reduce_sum(tf.multiply(self.x, tf.log(self.x_hat + 1e-6))
                                                + tf.multiply(1 - self.x, tf.log(1 - self.x_hat + 1e-6)))
            self.loss = (self.kl_loss / 2 + self.generate_loss) / self.batch_size

    def __create_summary(self):
        with tf.name_scope("summary"):
            self.summary_kl_loss = tf.summary.scalar('summary_kl_loss', self.kl_loss)
            self.summary_generate_loss = tf.summary.scalar('summary_generate_loss', self.generate_loss)
            self.summary_loss = tf.summary.scalar('summary_loss', self.loss)
            self.summary = tf.summary.merge([self.summary_kl_loss, self.summary_generate_loss, self.summary_loss],
                                            name='summary')

    def __create_optimizer(self):
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def build_network(self):
        self.__create_placeholder()
        self.__create_encoder()
        self.__create_latent()
        self.__create_decoder()
        self.__create_loss()
        self.__create_optimizer()
        self.__create_summary()


def train_model(model, train_data, num_epoch):
    saver = tf.train.Saver()
    if not os.path.exists('./model'):
        os.mkdir('./model')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('model/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        total_loss = 0
        writer = tf.summary.FileWriter('graph', sess.graph)
        initial_step = model.global_step.eval()
        num_batches = int(train_data.num_examples / model.batch_size)
        num_train_steps = num_batches * num_epoch
        num_step = 0
        for index in range(initial_step, initial_step + num_train_steps):
            x_batch, y_batch = train_data.next_batch(model.batch_size)
            feed_dict = {model.x: x_batch, model.y: y_batch}
            batch_loss, _, summary = sess.run([model.loss, model.optimizer, model.summary], feed_dict=feed_dict)
            writer.add_summary(summary, index)
            total_loss += batch_loss
            num_step += 1
            if (index + 1) % num_batches == 0:
                print('Average loss at step {}: {:5.1f}.'.format(index, total_loss / num_step))
                total_loss = 0
                num_step = 0
                saver.save(sess, 'model/VAE' + str(index))


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
            posterior, label, label_gt = sess.run([model.posterior, model.label, model.label_gt], feed_dict=feed_dict)
            y_gt += list(label_gt)
            y_pred += list(label)
    assignment = np.zeros([model.cluster_num, model.cluster_num])
    for i in range(len(y_pred)):
        assignment[y_pred[i]][y_gt[i]] += 1
    m = Munkres()
    indexes = m.compute(-assignment)
    total_cost = 0
    for row,col in indexes:
        total_cost += assignment[row][col]
    accuracy = total_cost / len(y_gt)
    print('{0} correct out of {1} samples. Accuracy = {2}.'.format(total_cost, len(y_gt), accuracy))


def main(dataset):
    model = VAEClustering(BATCH_SIZE, INPUT_DIM, LATENT_DIM_PER_CLUSTER, CLUSTER_NUM, LEARNING_RATE)
    model.build_network()
#    train_model(model, dataset.train, NUM_EPOCH)
    test_model(model, dataset.train)


if __name__ == '__main__':
    main(mnist)
