import tensorflow as tf
import time

import numpy as np
from PIL import Image

from tensorflow.examples.tutorials.mnist import input_data

import os

# change environment variables
os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '6'  # for the code to run on a old GPU (GTX-760)
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # use GPU 0

# set data and hyper parameters
mnist = input_data.read_data_sets("/data/mnist", one_hot=True)

NUM_EPOCH = 50
BATCH_SIZE = 100
LEARNING_RATE = 0.001
IMG_HEIGHT = 28
IMG_WIDTH = 28


# VAE Class
class VAE:
    """Build the graph for VAE"""
    def __init__(self, batch_size, img_height, img_width, input_dim,
                 hidden_dim, latent_dim, num_hidden, learning_rate):
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.input_dim = input_dim
        assert self.img_height * self.img_width == self.input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def __create_placeholders(self):
        """Create placeholders for input"""
        with tf.name_scope("data"):
            self.x = tf.placeholder(tf.float32, [self.batch_size, self.input_dim], 'x')

    def __create_encoder(self):
        """Create encoder network"""
        with tf.name_scope("encoder"):
            self.encoder_w = []
            self.encoder_b = []
            self.encoder = []
            self.encoder_activate = []
            self.previous_dim = self.input_dim
            self.previous_tensor = self.x
            for i_hidden in range(self.num_hidden):
                w_name = 'encoder_w' + str(i_hidden)
                self.encoder_w.append(tf.get_variable(w_name, [self.previous_dim, self.hidden_dim], tf.float32,
                                                      tf.contrib.layers.xavier_initializer(), trainable=True))
                b_name = 'encoder_b' + str(i_hidden)
                self.encoder_b.append(tf.Variable(tf.zeros([self.hidden_dim]), True, name=b_name))
                self.encoder.append(tf.matmul(self.previous_tensor, self.encoder_w[i_hidden]) + self.encoder_b[i_hidden])
                self.encoder_activate.append(tf.nn.tanh(self.encoder[i_hidden]))
                self.previous_dim = self.hidden_dim
                self.previous_tensor = self.encoder_activate[i_hidden]

    def __create_latent(self):
        """Create latent space"""
        with tf.name_scope("latent"):
            self.mu_z_w = tf.get_variable('mu_z_w', [self.previous_dim, self.latent_dim], tf.float32,
                                          tf.contrib.layers.xavier_initializer(), trainable=True)
            self.mu_z_b = tf.Variable(tf.zeros([self.latent_dim]), True, name='mu_z_b')
            self.mu_z = tf.matmul(self.previous_tensor, self.mu_z_w) + self.mu_z_b

            self.logsd_z_w = tf.get_variable('logsd_z_w', [self.previous_dim, self.latent_dim], tf.float32,
                                             tf.contrib.layers.xavier_initializer(), trainable=True)
            self.logsd_z_b = tf.Variable(tf.zeros([self.latent_dim]), True, name='logsd_z_b')
            self.logsd_z = tf.matmul(self.previous_tensor, self.logsd_z_w) + self.logsd_z_b
            self.sd_z = tf.exp(self.logsd_z, 'sd_z')

    def __create_decoder(self):
        """Create decoder (training procedure: random sample from N(mu_z, sd_z))"""
        with tf.name_scope("decoder"):
            self.noise = tf.random_normal([self.batch_size, self.latent_dim], name='noise')
            self.z = tf.multiply(self.noise, self.sd_z) + self.mu_z

            self.previous_dim = self.latent_dim
            self.previous_tensor = self.z
            self.decoder_w = []
            self.decoder_b = []
            self.decoder = []
            self.decoder_activate = []
            for i_hidden in range(self.num_hidden):
                w_name = 'decoder_w' + str(i_hidden)
                self.decoder_w.append(tf.get_variable(w_name, [self.previous_dim, self.hidden_dim], tf.float32,
                                                      tf.contrib.layers.xavier_initializer(), trainable=True))
                b_name = 'decoder_b' + str(i_hidden)
                self.decoder_b.append(tf.Variable(tf.zeros([self.hidden_dim]), True, name=b_name))
                self.decoder.append(tf.matmul(self.previous_tensor, self.decoder_w[i_hidden])
                                    + self.decoder_b[i_hidden])
                self.decoder_activate.append(tf.nn.tanh(self.decoder[i_hidden]))
                self.previous_dim = self.hidden_dim
                self.previous_tensor = self.decoder_activate[i_hidden]
            self.decoder_final_w = tf.get_variable('final_w', [self.previous_dim, self.input_dim], tf.float32,
                                                   tf.contrib.layers.xavier_initializer(), trainable=True)
            self.decoder_final_b = tf.Variable(tf.zeros([self.input_dim]), True, name='final_b')
            self.decoder_final = tf.matmul(self.previous_tensor, self.decoder_final_w) + self.decoder_final_b
            self.x_hat = tf.nn.sigmoid(self.decoder_final, 'x_hat')

    def __create_loss(self):
        """Create loss function"""
        with tf.name_scope("loss"):
            self.klloss_mu_z = tf.square(self.mu_z, 'klloss_mu_z')
            self.var_z = tf.square(self.sd_z, 'var_z')
            self.klloss_sd_z = self.var_z - 2 * self.logsd_z - 1
            self.klloss = tf.reduce_sum(self.klloss_mu_z + self.klloss_sd_z) / 2

            self.generate_loss = -tf.reduce_sum(tf.multiply(self.x, tf.log(self.x_hat))
                                                + tf.multiply(1 - self.x, tf.log(1 - self.x_hat)))

            self.loss = (self.klloss + self.generate_loss) / self.batch_size

    def __create_generate(self):
        """Create generate module"""
        with tf.name_scope('generate'):
            self.generate_decoder = []
            self.generate_decoder_activate = []
            self.previous_tensor = self.noise
            self.previous_dim = self.latent_dim
            for i_hidden in range(self.num_hidden):
                self.generate_decoder.append(tf.matmul(self.previous_tensor, self.decoder_w[i_hidden])
                                             + self.decoder_b[i_hidden])
                self.generate_decoder_activate.append(tf.nn.tanh(self.generate_decoder[i_hidden]))
                self.previous_dim = self.hidden_dim
                self.previous_tensor = self.generate_decoder_activate[i_hidden]
            self.generate_decoder_final = tf.matmul(self.previous_tensor, self.decoder_final_w) + self.decoder_final_b
            self.generate_x_hat = tf.reshape(tf.nn.sigmoid(self.generate_decoder_final, 'x_hat'),
                                             [self.batch_size, self.img_height, self.img_width])

    def __create_reconstruct(self):
        """Create reconstruct module"""
        with tf.name_scope('reconstruct'):
            self.reconstruct_decoder = []
            self.reconstruct_decoder_activate = []
            self.previous_tensor = self.mu_z
            self.previous_dim = self.latent_dim
            for i_hidden in range(self.num_hidden):
                self.reconstruct_decoder.append(tf.matmul(self.previous_tensor, self.decoder_w[i_hidden])
                                                + self.decoder_b[i_hidden])
                self.reconstruct_decoder_activate.append(tf.nn.tanh(self.reconstruct_decoder[i_hidden]))
                self.previous_dim = self.hidden_dim
                self.previous_tensor = self.reconstruct_decoder_activate[i_hidden]
            self.reconstruct_decoder_final = tf.matmul(self.previous_tensor, self.decoder_final_w) \
                                             + self.decoder_final_b
            self.reconstruct_x_hat = tf.nn.sigmoid(self.reconstruct_decoder_final)
#            self.reconstruct_error_l2 = tf.reduce_sum(tf.square(self.x - self.reconstruct_x_hat), 'error_l2') / self.batch_size
#            self.reconstruct_gt_l2 = tf.reduce_sum(tf.square(self.x), 'gt_l2') / self.batch_size

    def __create_optimizer(self):
        """Create optimizer"""
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def __create_summary(self):
        """Create summary"""
        with tf.name_scope('summary'):
            self.summary_loss = tf.summary.scalar('loss', self.loss)
            self.summary_generate_loss = tf.summary.scalar('generate_loss', self.generate_loss)
            self.summary_klloss = tf.summary.scalar('klloss', self.klloss)
            self.summary_train = tf.summary.merge([self.summary_loss, self.summary_generate_loss, self.summary_klloss])

            self.summary_generate = tf.summary.image('generate', self.generate_x_hat, 10)

    def build_graph(self):
        """Build the graph for VAE"""
        self.__create_placeholders()
        self.__create_encoder()
        self.__create_latent()
        self.__create_decoder()
        self.__create_loss()
        self.__create_generate()
        self.__create_reconstruct()
        self.__create_optimizer()
        self.__create_summary()


def train_model(model, train_data, test_data, num_epoch):
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
            feed_dict = {model.x: x_batch}
            batch_loss, _, summary = sess.run([model.loss, model.optimizer, model.summary_train], feed_dict=feed_dict)
            writer.add_summary(summary, index)
            total_loss += batch_loss
            num_step += 1
            if (index + 1) % num_batches == 0:
                print('Average loss at step {}: {:5.1f}'.format(index, total_loss / num_step))
                total_loss = 0
                num_step = 0
                saver.save(sess, 'model/VAE' + str(index))


def main(dataset):
    model = VAE(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 784, 200, 50, 2, LEARNING_RATE)
    model.build_graph()
    train_model(model, dataset.train, dataset.test, NUM_EPOCH)


if __name__ == '__main__':
    main(mnist)
