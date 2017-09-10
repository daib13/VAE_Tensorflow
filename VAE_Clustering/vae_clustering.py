import tensorflow as tf
import time

import numpy as np
from PIL import Image

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/data/mnist', one_hot=True)

import os
os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '6'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

batch_size = 100
learning_rate = 0.001
num_epoch = 200
num_cluster = 10
dim_z = 5

# build the network
x = tf.placeholder(tf.float32, [batch_size, 784], name='x')

encoder1_w = tf.get_variable(shape=[784, 200], trainable=True, name='encoder1_w', initializer=tf.contrib.layers.xavier_initializer())
encoder1_b = tf.Variable(tf.zeros([200]), trainable=True, name='encoder1_b')
encoder1 = tf.matmul(x, encoder1_w) + encoder1_b
encoder1_tanh = tf.nn.tanh(encoder1)

encoder2_w = tf.get_variable(shape=[200, 200], trainable=True, name='encoder2_w', initializer=tf.contrib.layers.xavier_initializer())
encoder2_b = tf.Variable(tf.zeros([200]), trainable=True, name='encoder2_b')
encoder2 = tf.matmul(encoder1, encoder2_w) + encoder2_b
encoder2_tanh = tf.nn.tanh(encoder2)

mu_z_w = tf.get_variable(shape=[200, dim_z], trainable=True, name='mu_z_w', initializer=tf.contrib.layers.xavier_initializer())
mu_z_b = tf.Variable(tf.zeros([dim_z]), trainable=True, name='mu_z_b')
mu_z = tf.matmul(encoder2_tanh, mu_z_w) + mu_z_b

logsd_z_w = tf.get_variable(shape=[200, dim_z], trainable=True, name='logsd_z_w', initializer=tf.contrib.layers.xavier_initializer())
logsd_z_b = tf.Variable(tf.zeros([dim_z]), trainable=True, name='logsd_z_b')
logsd_z = tf.matmul(encoder2_tanh, logsd_z_w) + logsd_z_b

sd_z = tf.exp(logsd_z)
var_z = tf.square(sd_z)

loss_mu_z = tf.square(mu_z)
loss_var_z = tf.subtract(var_z, 2*logsd_z) - 1
klloss = tf.reduce_sum(loss_mu_z + loss_var_z) / batch_size / 2

noise = tf.random_normal(shape=[batch_size, dim_z], dtype=tf.float32, name='noise')
sample = mu_z + tf.multiply(sd_z, noise)

decoder1_w = []
decoder1_b = []
decoder1 = []
for cluster_id in range(num_cluster):
    decoder1_w_name = 'decoder1_w' + str(cluster_id)
    decoder1_b_name = 'decoder1_b' + str(cluster_id)
    decoder1_w.append(tf.get_variable(shape=[dim_z, 200], trainable=True, name=decoder1_w_name, initializer=tf.contrib.layers.xavier_initializer()))
    decoder1_b.append(tf.Variable(tf.zeros([200]), trainable=True, name=decoder1_b_name))
    decoder1.append(tf.matmul(sample, decoder1_w[cluster_id]) + decoder1_b[cluster_id])
decoder1_concat = tf.concat(decoder1, 0)
decoder1_tanh_concat = tf.nn.tanh(decoder1_concat)

decoder2_w = tf.get_variable(shape=[200, 200], trainable=True, name='decoder2_w', initializer=tf.contrib.layers.xavier_initializer())
decoder2_b = tf.Variable(tf.zeros([200]), trainable=True, name='decoder2_b')
decoder2 = tf.matmul(decoder1_tanh_concat, decoder2_w) + decoder2_b
decoder2_tanh = tf.nn.tanh(decoder2)

decoder3_w = tf.get_variable(shape=[200, 784], trainable=True, name='decoder3_w', initializer=tf.contrib.layers.xavier_initializer())
decoder3_b = tf.Variable(tf.zeros([784]), trainable=True, name='decoder3_b')
decoder3 = tf.matmul(decoder2_tanh, decoder3_w) + decoder3_b
x_hat = tf.nn.sigmoid(decoder3)

x_hat_reshape = tf.reshape(x_hat, shape=[num_cluster, batch_size, 784])
x_reshape = tf.reshape(x, shape=[1, batch_size, 784])
x_duplicate = tf.tile(x_reshape, [num_cluster, 1, 1])

logit = x_duplicate * tf.log(x_hat_reshape + 1e-12) + (1 - x_duplicate) * tf.log(1 - x_hat_reshape + 1e-12)
log_likelihood = tf.reduce_sum(logit, 2)

prior_logit = tf.Variable(tf.zeros([num_cluster]), trainable=True, name='prior_logit')
prior = tf.nn.softmax(prior_logit, 0)
prior_reshape = tf.reshape(prior, [num_cluster, 1])
prior_duplicate = tf.tile(prior_reshape, [1, batch_size])
log_posterior = log_likelihood + tf.log(prior_duplicate)
posterior = tf.nn.softmax(log_posterior, dim=0)

generate_loss = tf.reduce_sum(-tf.multiply(log_likelihood, posterior)) / batch_size
prior_loss = tf.reduce_sum(tf.multiply(posterior, tf.log(posterior + 1e-6) - tf.log(prior_duplicate + 1e-6))) / batch_size

loss = klloss + generate_loss + prior_loss

# set the optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.00001).minimize(loss)

# start a session
sess = tf.Session()
writer = tf.summary.FileWriter('./graph', sess.graph)
writer.close()

# Train
start_time = time.time()
# 1) run with a random initialization
#sess.run(tf.global_variables_initializer())
# 2) run with a saved model
saver = tf.train.Saver()
saver.restore(sess, './model/model.ckpt')

num_batches = int(mnist.train.num_examples / batch_size)
for i in range(num_epoch):
    break
    total_loss = 0
    for batch_id in range(num_batches):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        _, loss_batch = sess.run([optimizer, loss], feed_dict={x: x_batch})
        sess.run(x, feed_dict={x: x_batch})
        klloss_val = klloss.eval(session=sess, feed_dict={x: x_batch})
        generate_loss_val = generate_loss.eval(session=sess, feed_dict={x: x_batch})
        prior_loss_val = prior_loss.eval(session=sess, feed_dict={x: x_batch})
        log_likelihood_val = log_likelihood.eval(session=sess, feed_dict={x: x_batch})
        total_loss += loss_batch
    print('Epoch = {0}, average loss = {1}.'.format(i, total_loss/num_batches))
print('Total time = {0}.'.format(time.time() - start_time))
print('Optimization finished.')

# Save model
saver = tf.train.Saver()
save_path = saver.save(sess, './model/model.ckpt')

# Test
decoder1_test = []
for cluster_id in range(10):
    decoder1_test.append(tf.matmul(noise, decoder1_w[cluster_id]) + decoder1_b[cluster_id])
decoder1_concat_test = tf.concat(decoder1_test, 0)
decoder1_tanh_concat_test = tf.nn.tanh(decoder1_concat_test)

decoder2_test = tf.matmul(decoder1_tanh_concat_test, decoder2_w) + decoder2_b
decoder2_tanh_test = tf.nn.tanh(decoder2_test)

decoder3_test = tf.matmul(decoder2_tanh_test, decoder3_w) + decoder3_b
x_hat_test = tf.nn.sigmoid(decoder3_test)

x_hat_reshape_test = tf.reshape(x_hat_test, [num_cluster, batch_size, 784])

sess.run(x_hat_reshape_test)
samples = x_hat_reshape_test.eval(session=sess)
for i in range(10):
    img = np.ones([300, 300], dtype=np.float32)
    for h in range(10):
        for w in range(10):
            img[30*h-29:30*h-1, 30*w-29:30*w-1] = np.reshape(samples[w, i*10 + h, :], [28, 28])
    rescaled = (255 * img).astype(np.uint8)
    im = Image.fromarray(rescaled)
    img_name = 'generate/' + str(i) + '.png'
    im.save(img_name)
