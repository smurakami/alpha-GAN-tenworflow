import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from model import encoder, generator, discriminator, code_discriminator
from tqdm import tqdm

import numpy as np
import cv2

# avoid log(0)
EPS = 1e-12

# Parameters
reconstruction_loss_weight = 40.0

# ======================================
#          Create Netrowks
# ======================================

images = tf.placeholder(tf.float32, (None, 32, 32, 1))
x_real = (images / 255.0) * 2 - 1

# --------------------------------------
#               Encoder

with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
    z_encoded = encoder(x_real)

# --------------------------------------
#               Generator

with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
    x_autoencoded = generator(z_encoded)

# sampled z
z_prior = tf.placeholder(tf.float32, (None, 128))

with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
    x_generated = generator(z_prior)

# --------------------------------------
#             Discriminator

# generated image
x_fake = tf.concat([x_autoencoded, x_generated], 0)

with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
    y_fake = discriminator(x_fake)

with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
    y_real = discriminator(x_real)

# --------------------------------------
#          Code Discriminator

with tf.variable_scope('code_discriminator', reuse=tf.AUTO_REUSE):
    c_real = code_discriminator(z_prior)

with tf.variable_scope('code_discriminator', reuse=tf.AUTO_REUSE):
    c_fake = code_discriminator(z_encoded)    


# ======================================
#             Define Loss
# ======================================

reconstruction_loss = reconstruction_loss_weight * tf.reduce_mean(tf.abs(x_real - x_autoencoded))
generator_loss = tf.reduce_mean(-tf.log(y_fake + EPS))
discriminator_loss = -tf.reduce_mean(tf.log(y_real + EPS)) - tf.reduce_mean(tf.log(1 - y_fake + EPS))
code_discriminator_loss = -tf.reduce_mean(tf.log(c_real + EPS)) - tf.reduce_mean(tf.log(1 - c_fake + EPS))
code_generator_loss = tf.reduce_mean(-tf.log(c_fake + EPS))

# loss for autoencoder
encoder_generator_loss = reconstruction_loss + generator_loss + code_generator_loss

# ======================================
#          Create Optimizer
# ======================================

variables =  tf.trainable_variables()
encoder_vars = [var for var in variables if 'encoder/' in var.name]
generator_vars = [var for var in variables if 'generator/' in var.name]
discriminator_vars = [var for var in variables if 'discriminator/' in var.name]
code_discriminator_vars = [var for var in variables if 'code_discriminator/' in var.name]


# This Parameter is important
lr = 8e-4
beta1= 0.5
beta2 = 0.9

# --------------------------------------
#             Auto Encoder

encoder_generator_opt = tf.train.AdamOptimizer(lr, beta1, beta2).minimize(
    encoder_generator_loss,
    var_list=encoder_vars + generator_vars)

# --------------------------------------
#             Discriminator

discriminator_opt = tf.train.AdamOptimizer(lr, beta1, beta2).minimize(
    discriminator_loss,
    var_list=discriminator_vars)

# --------------------------------------
#          Code Discriminator

code_discriminator_opt = tf.train.AdamOptimizer(lr, beta1, beta2).minimize(
    code_discriminator_loss,
    var_list=code_discriminator_vars)

data = np.load('data/cat_32.npy')


def generate_and_save_current_image(epoch):
    z_batch = np.random.uniform(-1, 1, (16, 128))
    generated = sess.run(x_generated, {z_prior: z_batch})
    rows = []
    for i in range(0, 16, 4):
        row = np.hstack(generated[i:i+4])
        rows.append(row)
    merged = np.vstack(rows)

    output = ((merged + 1) /2 * 255).astype(np.uint8)


    cv2.imwrite('images/output_{:03d}.png'.format(epoch), output)

# ======================================
#           Training Session
# ======================================

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 128
max_epoch = 2000
epoch = 0
saver = tf.train.Saver()

while epoch < max_epoch:
    for index in tqdm(range(0, len(data), batch_size)):
        image_batch = data[index:index+batch_size]
        sample_z = lambda: np.random.randn(len(image_batch), 128)
        # ----------TRAIN TWICE-----------------
        _, eg_loss = sess.run([encoder_generator_opt, encoder_generator_loss], {images: image_batch, z_prior: sample_z()})
        _, eg_loss = sess.run([encoder_generator_opt, encoder_generator_loss], {images: image_batch, z_prior: sample_z()}) # run twice
        # --------------------------------------
        _, d_loss = sess.run([discriminator_opt, discriminator_loss], {images: image_batch, z_prior: sample_z()})
        # --------------------------------------
        _, c_loss = sess.run([code_discriminator_opt, code_discriminator_loss], {images: image_batch, z_prior: sample_z()})
    saver.save(sess, './model/model', global_step=epoch)    
    print("epoch:", epoch+1, ", eg_loss:", eg_loss, ", d_loss:", d_loss, ", c_loss:", c_loss)
    generate_and_save_current_image(epoch)

    epoch += 1
