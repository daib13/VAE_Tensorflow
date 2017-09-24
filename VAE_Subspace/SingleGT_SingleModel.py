import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import os
os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '6'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

BATCH_SIZE = 100
LEARNING_RATE = 0.001
NUM_ITERATION = 500000

DIM_X = 3
DIM_Z = 2
DIM_HIDDEN = 20
NUM_HIDDEN = 2
EPSILON = 0.001

# Single GT
#noise_x = tf.random_normal([BATCH_SIZE, 1], dtype=tf.float32)
#x1 = tf.sin(noise_x)
#x2 = tf.cos(noise_x)
#x = tf.concat([x1, noise_x, x2], 1, 'x')

# Double GT
noise_x = tf.random_normal([int(BATCH_SIZE/2), 1], dtype=tf.float32)
x1 = tf.sin(noise_x)
x2 = tf.cos(noise_x)
x_part1 = tf.concat([x1, noise_x, x2], 1, 'x_part1')
x_part2 = tf.concat([-noise_x, x1, x2], 1, 'x_part2')
x = tf.concat([x_part1, x_part2], 0, 'x')

encoder1_w = tf.get_variable('encoder1_w', [DIM_X, DIM_HIDDEN], tf.float32,
                             tf.contrib.layers.xavier_initializer(), trainable=True)
encoder1_b = tf.Variable(tf.zeros([DIM_HIDDEN]), True, dtype=tf.float32, name='encoder1_b')
encoder1 = tf.matmul(x, encoder1_w) + encoder1_b
encoder1_tanh = tf.nn.relu(encoder1)

encoder2_w = tf.get_variable('encoder2_w', [DIM_HIDDEN, DIM_HIDDEN], tf.float32,
                             tf.contrib.layers.xavier_initializer(), trainable=True)
encoder2_b = tf.Variable(tf.zeros([DIM_HIDDEN]), True, dtype=tf.float32, name='encoder2_b')
encoder2 = tf.matmul(encoder1_tanh, encoder2_w) + encoder2_b
encoder2_tanh = tf.nn.relu(encoder2)

#encoder3_w = tf.get_variable('encoder3_w', [DIM_HIDDEN, DIM_HIDDEN], tf.float32,
#                             tf.contrib.layers.xavier_initializer(), trainable=True)
#encoder3_b = tf.Variable(tf.zeros([DIM_HIDDEN]), True, dtype=tf.float32, name='encoder3_b')
#encoder3 = tf.matmul(encoder2_tanh, encoder3_w) + encoder3_b
#encoder3_tanh = tf.nn.tanh(encoder3)

mu_z_w = tf.get_variable('mu_z_w', [DIM_HIDDEN, DIM_Z], tf.float32,
                         tf.contrib.layers.xavier_initializer(), trainable=True)
mu_z_b = tf.Variable(tf.zeros([DIM_Z]), True, dtype=tf.float32, name='mu_z_b')
mu_z = tf.matmul(encoder2_tanh, mu_z_w) + mu_z_b

logsd_z_w = tf.get_variable("logsd_z_w", [DIM_HIDDEN, DIM_Z], tf.float32,
                            tf.contrib.layers.xavier_initializer(), trainable=True)
logsd_z_b = tf.Variable(tf.zeros([DIM_Z]), True, dtype=tf.float32, name='logsd_z_b')
logsd_z = tf.matmul(encoder2_tanh, logsd_z_w) + logsd_z_b
sd_z = tf.exp(logsd_z)

sd_gt = tf.nn.softmax(tf.square(mu_z)) + 0.01

noise = tf.random_normal([BATCH_SIZE, DIM_Z], dtype=tf.float32, name='noise')
z = tf.multiply(noise, sd_z) + mu_z

decoder1_w = tf.get_variable('decoder1_w', [DIM_Z, DIM_HIDDEN], tf.float32,
                             tf.contrib.layers.xavier_initializer(), trainable=True)
decoder1_b = tf.Variable(tf.zeros([DIM_HIDDEN]), True, dtype=tf.float32, name='decoder1_b')
decoder1 = tf.matmul(z, decoder1_w) + decoder1_b
decoder1_tanh = tf.nn.relu(decoder1)

#decoder2_w = tf.get_variable('decoder2_w', [DIM_HIDDEN, DIM_HIDDEN], tf.float32,
#                             tf.contrib.layers.xavier_initializer(), trainable=True)
#decoder2_b = tf.Variable(tf.zeros([DIM_HIDDEN]), True, dtype=tf.float32, name='decoder2_b')
#decoder2 = tf.matmul(decoder1_tanh, decoder2_w) + decoder2_b
#decoder2_tanh = tf.nn.tanh(decoder2)

#decoder3_w = tf.get_variable('decoder3_w', [DIM_HIDDEN, DIM_HIDDEN], tf.float32,
#                             tf.contrib.layers.xavier_initializer(), trainable=True)
#decoder3_b = tf.Variable(tf.zeros([DIM_HIDDEN]), True, dtype=tf.float32, name='decoder3_b')
#decoder3 = tf.matmul(decoder2_tanh, decoder3_w) + decoder3_b
#decoder3_tanh = tf.nn.tanh(decoder3)

decoder_final_w = tf.get_variable('decoder_final_w', [DIM_HIDDEN, DIM_X], tf.float32,
                                  tf.contrib.layers.xavier_initializer(), trainable=True)
decoder_final_b = tf.Variable(tf.zeros([DIM_X]), True, dtype=tf.float32, name='decoder_final_b')
x_hat = tf.matmul(decoder1_tanh, decoder_final_w) + decoder_final_b

klloss_mu_z = tf.square(tf.divide(mu_z, sd_gt), 'klloss_mu_z')
var_z = tf.square(tf.divide(sd_z, sd_gt), 'var_z')
klloss_sd_z = var_z - 2 * logsd_z - 1
klloss = tf.reduce_sum(klloss_mu_z + klloss_sd_z)
#generate_loss = tf.reduce_sum(tf.log(tf.square(x - x_hat) + EPSILON))
generate_loss = tf.reduce_sum(tf.square(x - x_hat))
loss = (0.1*klloss + generate_loss) / 2 / BATCH_SIZE
preloss = (generate_loss + 0.1 * klloss) / 2 / BATCH_SIZE

preoptimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(preloss)
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
summary_loss = tf.summary.scalar('loss', loss)

gen_decoder1 = tf.matmul(noise, decoder1_w) + decoder1_b
gen_decoder1_tanh = tf.nn.relu(gen_decoder1)
#gen_decoder2 = tf.matmul(gen_decoder1, decoder2_w) + decoder2_b
#gen_decoder2_tanh = tf.nn.tanh(gen_decoder2)
#gen_decoder3 = tf.matmul(gen_decoder2, decoder3_w) + decoder3_b
#gen_decoder3_tanh = tf.nn.tanh(gen_decoder3)
gen_x_hat = tf.matmul(gen_decoder1_tanh, decoder_final_w) + decoder_final_b

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('graph', sess.graph)
#    for index in range(1000):
#        _ = sess.run(preoptimizer)
    for index in range(NUM_ITERATION):
        total_loss, _, summary = sess.run([loss, optimizer, summary_loss])
        if index % 1000 == 0:
            writer.add_summary(summary, index)
            print('iter = {0}， loss = {1}.'.format(index, total_loss))

        if index % 10000 == 0:
            fig2 = plt.figure(2)
            ax12 = fig2.add_subplot(121, projection='3d')
            ax22 = fig2.add_subplot(122)
            for i in range(10):
                gen, gt, gen_z, gt_z = sess.run([gen_x_hat, x, noise, mu_z])
                ax12.scatter(gen[:, 0], gen[:, 1], gen[:, 2], c='r', marker='o')
                ax12.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c='g', marker='o')
                ax22.scatter(gen_z[:, 0], gen_z[:, 1], c='r', marker='o')
                ax22.scatter(gt_z[:, 0], gt_z[:, 1], c='g', marker='o')
            plt.savefig(str(index) + '.png')

    w = decoder1_w.eval(session=sess)
    for d in range(DIM_Z):
        l2 = np.linalg.norm(w[d, :])
        print(l2)

    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    for i in range(10):
        gen, gt, gen_z, gt_z = sess.run([gen_x_hat, x, noise, mu_z])
        ax1.scatter(gen[:, 0], gen[:, 1], gen[:, 2], c='r', marker='o')
        ax1.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c='g', marker='o')
        ax2.scatter(gen_z[:, 0], gen_z[:, 1], c='r', marker='o')
        ax2.scatter(gt_z[:, 0], gt_z[:, 1], c='g', marker='o')
    plt.show()
