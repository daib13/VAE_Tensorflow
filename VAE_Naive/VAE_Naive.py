import tensorflow as tf
import time

import numpy as np
from PIL import Image

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/data/mnist", one_hot=True)
num_epoch = 50

import os
# change environment variables
os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '6'  # for the code to run on a old GPU (GTX-760)
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # use GPU 0

# set hyper parameters
batch_size = 100
learning_rate = 0.001

# build the network
# Network structure: 784-200-200-50-200-200-784 (activation: TanH)
x = tf.placeholder(tf.float32, [batch_size, 784], name='x')

# encoder
encoder1_w = tf.get_variable(shape=[784, 200], trainable=True, name='encoder1_w', initializer=tf.contrib.layers.xavier_initializer())
encoder1_b = tf.Variable(tf.zeros([200]), trainable=True, name='encoder1_b')
encoder1 = tf.matmul(x, encoder1_w) + encoder1_b
encoder1_tanh = tf.nn.tanh(encoder1)

encoder2_w = tf.get_variable(shape=[200, 200], trainable=True, name='encoder2_w', initializer=tf.contrib.layers.xavier_initializer())
encoder2_b = tf.Variable(tf.zeros([200]), trainable=True, name='encoder2_b')
encoder2 = tf.matmul(encoder1_tanh, encoder2_w) + encoder2_b
encoder2_tanh = tf.nn.tanh(encoder2)

# mu_z and sd_z share the same encoder
mu_z_w = tf.get_variable(shape=[200, 50], trainable=True, name='mu_z_w', initializer=tf.contrib.layers.xavier_initializer())
mu_z_b = tf.Variable(tf.zeros([50]), trainable=True, name='mu_z_b')
mu_z = tf.matmul(encoder2_tanh, mu_z_w) + mu_z_b

logsd_z_w = tf.get_variable(shape=[200, 50], trainable=True, name='logsd_z_w', initializer=tf.contrib.layers.xavier_initializer())
logsd_z_b = tf.Variable(tf.zeros([50]), trainable=True, name='logsd_z_b')
logsd_z = tf.matmul(encoder2_tanh, logsd_z_w) + logsd_z_b

sd_z = tf.exp(logsd_z)
var_z = tf.square(sd_z)

# klloss
loss_mu_z = tf.square(mu_z)
loss_var_z = tf.subtract(var_z, 2 * logsd_z) - 1
klloss = tf.reduce_sum(loss_mu_z + loss_var_z) / batch_size / 2

# random sample
noise = tf.random_normal(shape=[batch_size, 50], dtype=tf.float32, name='noise')
z = mu_z + tf.multiply(sd_z, noise)

# decoder
decoder1_w = tf.get_variable(shape=[50, 200], trainable=True, name='decoder1_w', initializer=tf.contrib.layers.xavier_initializer())
decoder1_b = tf.Variable(tf.zeros([200]), trainable=True, name='decoder1_b')
decoder1 = tf.matmul(z, decoder1_w) + decoder1_b
decoder1_tanh = tf.nn.tanh(decoder1)

decoder2_w = tf.get_variable(shape=[200, 200], trainable=True, name='decoder2_w', initializer=tf.contrib.layers.xavier_initializer())
decoder2_b = tf.Variable(tf.zeros([200]), trainable=True, name='decoder2_b')
decoder2 = tf.matmul(decoder1_tanh, decoder2_w) + decoder2_b
decoder2_tanh = tf.nn.tanh(decoder2)

decoder3_w = tf.get_variable(shape=[200, 784], trainable=True, name='decoder3_w', initializer=tf.contrib.layers.xavier_initializer())
decoder3_b = tf.Variable(tf.zeros([784]), trainable=True, name='decoder3_b')
decoder3 = tf.matmul(decoder2_tanh, decoder3_w) + decoder3_b
x_hat = tf.nn.sigmoid(decoder3)

# Bournelli distribution
generate_loss = -tf.reduce_sum(x * tf.log(x_hat) + (1-x) * tf.log(1-x_hat)) / batch_size
loss = klloss + generate_loss

# Set the optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.00001).minimize(loss)

# Start a session
sess = tf.Session()
writer = tf.summary.FileWriter('./graph', sess.graph)
writer.close()

# Train
start_time = time.time()
# 1) run with a random initialization
sess.run(tf.global_variables_initializer())
# 2) run with a saved model (if you want to use this option, make sure there is a model saved in the path)
#saver = tf.train.Saver()
#saver.restore(sess, './model/model.ckpt')

num_batches = int(mnist.train.num_examples / batch_size)
for i in range(num_epoch):
    total_loss = 0
    for batch_id in range(num_batches):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        _, loss_batch = sess.run([optimizer, loss], feed_dict={x: x_batch})
        total_loss += loss_batch
    print('Epoch = {0}, average loss = {1}.'.format(i, total_loss/num_batches))
print('Total time = {0}.'.format(time.time() - start_time))
print('Optimization finished.')

# Save model
saver = tf.train.Saver()
save_path = saver.save(sess, './model/model.ckpt')

# Test (generate samples)
decoder1_test = tf.matmul(noise, decoder1_w) + decoder1_b
decoder1_tanh_test = tf.nn.tanh(decoder1_test)

decoder2_test = tf.matmul(decoder1_tanh_test, decoder2_w) + decoder2_b
decoder2_tanh_test = tf.nn.tanh(decoder2_test)

decoder3_test = tf.matmul(decoder2_tanh_test, decoder3_w) + decoder3_b
x_hat_test = tf.nn.sigmoid(decoder3_test)

# Random generate 10 samples
if not os.path.exists('./generate'):
    os.mkdir('./generate')
for i in range(10):
    sess.run(x_hat_test)
    samples = x_hat_test.eval(session=sess)
    img = np.ones([300, 300], dtype=np.float32)
    for h in range(10):
        for w in range(10):
            img[30*h-29:30*h-1, 30*w-29:30*w-1] = np.reshape(samples[h*10+w-10, :], [28, 28])
    rescaled = (255 * img).astype(np.uint8)
    im = Image.fromarray(rescaled)
    img_name = './generate/' + str(i) + '.png'
    im.save(img_name)
