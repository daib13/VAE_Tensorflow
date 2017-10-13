import tensorflow as tf
from tensorflow.contrib import layers
import os
import sys
import numpy as np
import data_util as du
from munkres import Munkres, print_matrix
import math

DIM_LAYER = [784, 500, 500, 2000, 10]
NUM_CLUSTER = 10


class SaeNet:
    class SubSaeNet:
        def __init__(self, input_type, input_dim, hidden_type, hidden_dim, parent_net, previous_net=None):
            self.protector = parent_net.protector
            self.batch_size = parent_net.batch_size
            if previous_net:
                self.depth = previous_net.depth + 1
            else:
                self.depth = 1
            with tf.name_scope('Depth' + str(self.depth)):
                self.global_step = tf.Variable(0, False, name='global_step', dtype=tf.int32)
                if previous_net:
                    self.input = tf.stop_gradient(previous_net.hidden)
                else:
                    self.input = parent_net.x
                with tf.variable_scope('Depth' + str(self.depth)):
                    self.encoder_w = tf.get_variable('encoder_w', [input_dim, hidden_dim],
                                                     tf.float32, layers.xavier_initializer(), trainable=True)
                    self.encoder_b = tf.get_variable('encoder_b', [hidden_dim], tf.float32,
                                                     tf.zeros_initializer(), trainable=True)
                    self.decoder_w = tf.get_variable('decoder_w', [hidden_dim, input_dim],
                                                     tf.float32, layers.xavier_initializer(), trainable=True)
                    self.decoder_b = tf.get_variable('decoder_b', [input_dim], tf.float32,
                                                     tf.zeros_initializer(), trainable=True)
                with tf.name_scope('Encoder'):
                    if hidden_type == 'free':
                        self.hidden = tf.matmul(self.input, self.encoder_w) + self.encoder_b
                    elif hidden_type == 'relu':
                        self.hidden = tf.nn.relu(tf.matmul(self.input, self.encoder_w) + self.encoder_b)
                    elif hidden_type == 'tanh':
                        self.hidden = tf.nn.tanh(tf.matmul(self.input, self.encoder_w) + self.encoder_b)
                    elif hidden_type == 'sigmoid':
                        self.hidden = tf.nn.sigmoid(tf.matmul(self.input, self.encoder_w) + self.encoder_b)
                with tf.name_scope('Decoder'):
                    if input_type == 'free':
                        self.output = tf.matmul(self.hidden, self.decoder_w) + self.decoder_b
                    elif input_type == 'relu':
                        self.output = tf.nn.relu(tf.matmul(self.hidden, self.decoder_w) + self.decoder_b)
                    elif input_type == 'tanh':
                        self.output = tf.nn.tanh(tf.matmul(self.hidden, self.decoder_w) + self.decoder_b)
                    elif input_type == 'sigmoid':
                        self.output = tf.nn.sigmoid(tf.matmul(self.hidden, self.decoder_w) + self.decoder_b)
                with tf.name_scope('Loss'):
                    if input_type == 'sigmoid':
                        self.loss = (-self.input * tf.log(self.output + self.protector)
                                     - (1 - self.input) * (tf.log(1 - self.output + self.protector))) / self.batch_size
                    else:
                        self.loss = tf.square(self.input - self.output) / self.batch_size
                with tf.name_scope('Optimizer'):
                    self.optimizer = tf.train.AdamOptimizer(parent_net.learning_rate,
                                                            epsilon=parent_net.epsilon).minimize(self.loss,
                                                                                                 self.global_step)
                with tf.name_scope('Summary'):
                    self.summary = tf.summary.scalar('loss', self.loss)

    def __init__(self, dim_layer, batch_size, epsilon, data_type):
        self.dim_layer = dim_layer
        self.batch_size = batch_size
        self.num_hidden_layer = len(dim_layer) - 1
        for i in range(self.num_hidden_layer + 1):
            assert dim_layer[i] > 0
        self.data_type = data_type
        self.protector = 1e-20
        self.epsilon = epsilon
        self.global_step = tf.Variable(0, False, dtype=tf.int32, name='global_step')

    def __create_layerwise_net(self):
        with tf.name_scope('SAE'):
            self.x = tf.placeholder(tf.float32, [self.batch_size, self.dim_layer[0]], 'x')
        with tf.name_scope('LearningRate'):
            self.learning_rate = tf.placeholder(tf.float32, [], 'learning_rate')
        self.sub_net = []
        for i in range(self.num_hidden_layer):
            if i == 0:
                self.input_type = self.data_type
                self.previous_net = None
            else:
                self.input_type = 'relu'
                self.previous_net = self.sub_net[i-1]
            if i == self.num_hidden_layer - 1:
                self.hidden_type = 'free'
            else:
                self.hidden_type = 'relu'
            self.sub_net.append(self.SubSaeNet(self.input_type, self.dim_layer[i],
                                               self.hidden_type, self.dim_layer[i + 1],
                                               self, self.previous_net))

    def __create_net(self):
        with tf.name_scope('SAE'):
            self.previous_tensor = self.x
            with tf.name_scope('Encoder'):
                self.encoder = []
                for i in range(self.num_hidden_layer):
                    with tf.variable_scope('Depth' + str(i + 1), reuse=True):
                        self.w = tf.get_variable('encoder_w')
                        self.b = tf.get_variable('encoder_b')
                    if i == self.num_hidden_layer - 1:
                        self.encoder.append(tf.matmul(self.previous_tensor, self.w) + self.b)
                    else:
                        self.encoder.append(tf.nn.relu(tf.matmul(self.previous_tensor, self.w) + self.b))
                    self.previous_tensor = self.encoder[i]
            with tf.name_scope('Decoder'):
                self.decoder = []
                for i in range(self.num_hidden_layer):
                    with tf.variable_scope('Depth' + str(self.num_hidden_layer - i), reuse=True):
                        self.w = tf.get_variable('decoder_w')
                        self.b = tf.get_variable('decoder_b')
                    if i == self.num_hidden_layer - 1:
                        if self.data_type == 'free':
                            self.decoder.append(tf.matmul(self.previous_tensor, self.w) + self.b)
                        elif self.data_type == 'sigmoid':
                            self.decoder.append(tf.nn.sigmoid(tf.matmul(self.previous_tensor, self.w)) + self.b)
                    else:
                        self.decoder.append(tf.nn.relu(tf.matmul(self.previous_tensor, self.w)) + self.b)
                    self.previous_tensor = self.decoder[i]
            with tf.name_scope('Loss'):
                if self.data_type == 'sigmoid':
                    self.loss = (- self.x * tf.log(self.decoder[self.num_hidden_layer-1] + self.protector)
                                 - (1 - self.x) * tf.log(1 - self.decoder[self.num_hidden_layer-1] + self.protector)) \
                                / self.batch_size
                else:
                    self.loss = tf.square(self.x - self.decoder[self.num_hidden_layer-1]) / self.batch_size
            with tf.name_scope('Optimizer'):
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate,
                                                        epsilon=self.epsilon).minimize(self.loss,
                                                                                       self.global_step)
            with tf.name_scope('Summary'):
                self.summary = tf.summary.scalar('loss', self.loss)

    def build_graph(self):
        self.__create_layerwise_net()
        self.__create_net()


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


def train_model(model, x, y):
    saver = tf.train.Saver()
    if not os.path.exists('./model'):
        os.mkdir('./model')

    with tf.Session() as sess:
        # initialize
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('model/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        # train stack autoencoder
        writer = tf.summary.FileWriter('graph', sess.graph)
        iteration_per_epoch = int(math.floor(x.shape[0] / model.batch_size))

        for i in range(model.num_hidden_layer):
            for epoch in range(10):
                x, y = du.shuffle(x, y)
                total_loss = 0
                lr = 0.002
                for index in range(iteration_per_epoch):
                    start_idx = index * model.batch_size
                    end_idx = start_idx + model.batch_size
                    feed_dict = {model.x: x[start_idx:end_idx, :], model.learning_rate: lr}
                    submodel = model.sub_net[i]
                    batch_loss, _, summary = sess.run([submodel.loss, submodel.optimizer, submodel.summary],
                                                      feed_dict=feed_dict)
                    total_loss += batch_loss
                    writer.add_summary(summary, index)
                total_loss /= iteration_per_epoch
                print('Layer = {0}, epoch = {1}, loss = {2}.'.format(i, epoch, total_loss))
                saver.save(sess, 'model/SAE' + str(i) + '_' + str(epoch))


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

    images_train, labels_train = du.load_mnist_data('training')
#    images_test, labels_test = load_mnist_data('testing')

    model = SaeNet(DIM_LAYER, 100, 0.0001, 'sigmoid')
    model.build_graph()

    train_model(model, images_train, labels_train)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
