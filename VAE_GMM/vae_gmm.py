import tensorflow as tf
from tensorflow.contrib import layers
import os
import sys
import numpy as np
from mnist import MNIST
from munkres import Munkres, print_matrix

class VAE_GMM:
    def __init__(self, batch_size, input_dim,
                 num_cluster, learning_rate, epsilon):
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.num_cluster = num_cluster
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, False, dtype=tf.int32, name='global_step')
        self.epsilon = epsilon

    def __create_encoder(self):
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [self.batch_size, self.input_dim], 'x')
            self.y = tf.placeholder(tf.float32, [self.batch_size, 10], 'y')
        with tf.name_scope('encoder'):
            with tf.name_scope('hidden1'):
                self.encoder_w1 = tf.get_variable('encoder_w1', [self.input_dim, 500], tf.float32,
                                                  layers.xavier_initializer(), trainable=True)
                self.encoder_b1 = tf.Variable(tf.zeros([500]), True, name='encoder_b1', dtype=tf.float32)
                self.encoder1 = tf.matmul(self.x, self.encoder_w1) + self.encoder_b1
                self.encoder_activate1 = tf.nn.relu(self.encoder1, 'hidden1')
            with tf.name_scope('hidden2'):
                self.encoder_w2 = tf.get_variable('encoder_w2', [500, 500], tf.float32,
                                                  layers.xavier_initializer(), trainable=True)
                self.encoder_b2 = tf.Variable(tf.zeros([500]), True, name='encoder_b2', dtype=tf.float32)
                self.encoder2 = tf.matmul(self.encoder_activate1, self.encoder_w2) + self.encoder_b2
                self.encoder_activate2 = tf.nn.relu(self.encoder2, 'hidden2')
            with tf.name_scope('hidden3'):
                self.encoder_w3 = tf.get_variable('encoder_w3', [500, 2000], tf.float32,
                                                  layers.xavier_initializer(), trainable=True)
                self.encoder_b3 = tf.Variable(tf.zeros([2000]), True, name='encoder_b3', dtype=tf.float32)
                self.encoder3 = tf.matmul(self.encoder_activate2, self.encoder_w3) + self.encoder_b3
                self.encoder_activate3 = tf.nn.relu(self.encoder3, 'hidden3')
            with tf.name_scope('mu_z'):
                self.mu_z_w = tf.get_variable('mu_z_w', [2000, 10], tf.float32,
                                              layers.xavier_initializer(), trainable=True)
                self.mu_z_b = tf.Variable(tf.zeros([10]), True, name='mu_z_b', dtype=tf.float32)
                self.mu_z = tf.matmul(self.encoder_activate3, self.mu_z_w) + self.mu_z_b
            with tf.name_scope('sd_z'):
                self.logsd_z_w = tf.get_variable('logsd_z_w', [2000, 10], tf.float32, 
                                              layers.xavier_initializer(), trainable=True)
                self.logsd_z_b = tf.Variable(tf.zeros([10]), True, name='logsd_z_b', dtype=tf.float32)
                self.logsd_z = tf.matmul(self.encoder_activate3, self.logsd_z_w) + self.logsd_z_b
                self.sd_z = tf.exp(self.logsd_z, 'sd_z')
            
    def __create_decoder(self):
        with tf.name_scope('sampling'):
            self.noise = tf.random_normal([self.batch_size, 10], dtype=tf.float32, name='noise')
            self.z = tf.multiply(self.sd_z, self.noise) + self.mu_z
        with tf.name_scope('decoder'):
            with tf.name_scope('hidden1'):
                self.decoder_w1 = tf.get_variable('decoder_w1', [10, 2000], tf.float32,
                                                  layers.xavier_initializer(), trainable=True)
                self.decoder_b1 = tf.Variable(tf.zeros([2000]), True, name='decoder_b1', dtype=tf.float32)
                self.decoder1 = tf.matmul(self.z, self.decoder_w1) + self.decoder_b1
                self.decoder_activate1 = tf.nn.relu(self.decoder1, 'hidden1')
            with tf.name_scope('hidden2'):
                self.decoder_w2 = tf.get_variable('decoder_w2', [2000, 500], tf.float32,
                                                  layers.xavier_initializer(), trainable=True)
                self.decoder_b2 = tf.Variable(tf.zeros([500]), True, name='decoder_b2', dtype=tf.float32)
                self.decoder2 = tf.matmul(self.decoder_activate1, self.decoder_w2) + self.decoder_b2
                self.decoder_activate2 = tf.nn.relu(self.decoder2)
            with tf.name_scope('hidden3'):
                self.decoder_w3 = tf.get_variable('decoder_w3', [500, 500], tf.float32,
                                                  layers.xavier_initializer(), trainable=True)
                self.decoder_b3 = tf.Variable(tf.zeros([500]), True, name='decoder_b3', dtype=tf.float32)
                self.decoder3 = tf.matmul(self.decoder_activate2, self.decoder_w3) + self.decoder_b3
                self.decoder_activate3 = tf.nn.relu(self.decoder3)
            with tf.name_scope('x_hat'):
                self.x_hat_w = tf.get_variable('x_hat_w', [500, self.input_dim], tf.float32,
                                               layers.xavier_initializer(), trainable=True)
                self.x_hat_b = tf.Variable(tf.zeros([self.input_dim]), True, name='x_hat_b', dtype=tf.float32)
                self.x_hat_net = tf.matmul(self.decoder_activate3, self.x_hat_w) + self.x_hat_b
                self.x_hat = tf.nn.sigmoid(self.x_hat_net, 'x_hat')
                
    def __create_loss(self):
        with tf.name_scope('loss'):
            with tf.name_scope('generate'):
                self.p_x_z_logit = self.x * tf.log(self.x_hat + 1e-12) + (1 - self.x) * tf.log(1 - self.x_hat + 1e-12)
                self.generate_loss = tf.reduce_sum(-self.p_x_z_logit) / self.batch_size
            with tf.name_scope('kl'):
                self.mu_c = tf.Variable(tf.random_normal([self.num_cluster, 10], 0, 0.1), True, name='mu_c', dtype=tf.float32)
                self.logsd_c = tf.Variable(tf.zeros([self.num_cluster, 10]), True, name='logsd_c', dtype=tf.float32)
                self.sd_c = tf.exp(self.logsd_c)
                self.prior_logit = tf.Variable(tf.zeros([10]), False, name='prior_logit', dtype=tf.float32)
                self.prior = tf.nn.softmax(self.prior_logit, name='prior')
                with tf.name_scope('prior'):
                    self.q_logit = tf.reduce_sum(-self.logsd_z - tf.square(self.noise) / 2, 1)
                with tf.name_scope('posterior'):
                    self.prior_logit_tile = tf.tile(tf.reshape(tf.log(self.prior), [1, 10]), [self.batch_size, 1])
                    self.z_tile = tf.tile(tf.reshape(self.z, [self.batch_size, 1, 10]), [1, self.num_cluster, 1])
                    self.mu_c_tile = tf.tile(tf.reshape(self.mu_c, [1, self.num_cluster, 10]), [self.batch_size, 1, 1])
                    self.sd_c_tile = tf.tile(tf.reshape(self.sd_c, [1, self.num_cluster, 10]), [self.batch_size, 1, 1])
                    self.logsd_c_tile = tf.tile(tf.reshape(self.logsd_c, [1, self.num_cluster, 10]),
                                                [self.batch_size, 1, 1])
                    self.p_logit = tf.reduce_sum(-self.logsd_c_tile 
                                                 - tf.square((self.z_tile - self.mu_c_tile) / self.sd_c_tile), 2) \
                                   + self.prior_logit_tile
                    self.p_logit_max = tf.reduce_max(self.p_logit, 1)
                    self.p_logit_max_tile = tf.tile(tf.reshape(self.p_logit_max, [self.batch_size, 1]), [1, 10])
                    self.p_res_sum = tf.reduce_sum(tf.exp(self.p_logit - self.p_logit_max_tile), 1)
                    self.p_logit_sum = self.p_logit_max + tf.log(self.p_res_sum)
                    self.posterior = tf.nn.softmax(self.p_logit, 1, name='posterior')
                self.kl_loss = tf.reduce_sum(self.q_logit - self.p_logit_sum) / self.batch_size
            self.loss = self.generate_loss + self.kl_loss
        with tf.name_scope('label'):
            self.label = tf.arg_max(self.posterior, 1, name='label')
            self.label_gt = tf.arg_max(self.y, 1, name='label_gt')
        
    def __create_optimizer(self):
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=self.epsilon).minimize(self.loss, self.global_step)
            
    def __create_summary(self):
        with tf.name_scope('summary'):
            self.summary_kl_loss = tf.summary.scalar('kl_loss', self.kl_loss)
            self.summary_generate_loss = tf.summary.scalar('generate_loss', self.generate_loss)
            self.summary_loss = tf.summary.scalar('loss', self.loss)
            self.summary_label = tf.summary.histogram('label', self.label)
            self.summary = tf.summary.merge([self.summary_kl_loss, self.summary_generate_loss, 
                                             self.summary_loss, self.summary_label], name='summary')
                
    def build_graph(self):
        self.__create_encoder()
        self.__create_decoder()
        self.__create_loss()
        self.__create_optimizer()
        self.__create_summary()
        

def load_mnist_data(flag='training'):
    mndata = MNIST('../data/MNIST')
    try:
        if flag == 'training':
            images, labels = mndata.load_training()
        elif flag == 'testing':
            images, labels = mndata.load_testing()
        else:
            raise Exception('Flag should be either training or testing.')
    except Exception:
        print("Flag error")
        raise
    images_array = np.array(images) / 255
    labels_array = np.array(labels)
    one_hot_labels = np.zeros((labels_array.size, labels_array.max() + 1))
    one_hot_labels[np.arange(labels_array.size), labels_array] = 1
    return images_array, one_hot_labels


def train_model(model, x, y, num_epoch):
    saver = tf.train.Saver()
    if not os.path.exists('./model'):
        os.mkdir('./model')

    with tf.Session() as sess:
        # initialize
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('model/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        # train autoencoder
        total_loss = 0
        writer = tf.summary.FileWriter('graph', sess.graph)
        initial_step = model.global_step.eval()
        iteration_per_epoch = x.shape[0] / model.batch_size
        finish_step = num_epoch * iteration_per_epoch
        start_idx = 0
        epoch_id = 0
        end_idx = start_idx + model.batch_size
        for index in range(initial_step, int(finish_step)):
            feed_dict = {model.x: x[start_idx:end_idx, :], model.y: y[start_idx:end_idx]}
            start_idx += model.batch_size
            end_idx += model.batch_size
            if end_idx >= x.shape[0]:
                start_idx = 0
                end_idx = start_idx + model.batch_size
                epoch_id += 1
                if epoch_id % 10 == 0:
                    model.learning_rate *= 0.9
            batch_loss, _, summary = sess.run([model.loss, model.optimizer, model.summary], feed_dict=feed_dict)
            total_loss += batch_loss
            writer.add_summary(summary, index)

            if (index + 1) % iteration_per_epoch == 0:
                print('Iter = {0}, loss = {1}.'.format(index, total_loss / iteration_per_epoch))
                total_loss = 0

            if (index + 1) % iteration_per_epoch == 0:
                saver.save(sess, 'model/VAE' + str(index))


def test_model(model, x, y):
    y_pred = list()
    y_gt = list()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('./model/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        iteration_per_epoch = int(x.shape[0] / model.batch_size)
        start_idx = 0
        end_idx = start_idx + model.batch_size
        for index in range(iteration_per_epoch):
            feed_dict = {model.x: x[start_idx:end_idx, :], model.y: y[start_idx:end_idx]}
            start_idx += model.batch_size
            end_idx += model.batch_size
            posterior, label, label_gt = sess.run([model.posterior, model.label, model.label_gt], feed_dict=feed_dict)
            y_gt += list(label_gt)
            y_pred += list(label)
    assignment = np.zeros([10, 10])
    for i in range(len(y_pred)):
        assignment[y_pred[i]][y_gt[i]] += 1
    m = Munkres()
    indexes = m.compute(-assignment)
    total_cost = 0
    for row,col in indexes:
        total_cost += assignment[row][col]
    accuracy = total_cost / len(y_gt)
    print('{0} correct out of {1} samples. Accuracy = {2}.'.format(total_cost, len(y_gt), accuracy))


def main():
    NUM_EPOCH = 300

    images_train, labels_train = load_mnist_data('training')
    images_test, labels_test = load_mnist_data('testing')

    model = VAE_GMM(100, 784, 10, 0.002, 0.00001)
    model.build_graph()

    train_model(model, images_train, labels_train, NUM_EPOCH)
    test_model(model, images_train, labels_train)
    test_model(model, images_test, labels_test)


if __name__ == '__main__':
    main()
