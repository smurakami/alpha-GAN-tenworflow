import tensorflow as tf


def res_block(inputs, filters, kernel_size, strides=(1, 1), activation=tf.nn.relu, kernel_initializer=None):
    x = inputs

    x = tf.layers.conv2d(x, filters, kernel_size=kernel_size, strides=strides, padding="same", kernel_initializer=kernel_initializer)
    x = tf.layers.batch_normalization(x, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=kernel_initializer)
    x = activation(x)

    x = tf.layers.conv2d(x, filters, kernel_size=kernel_size, strides=strides, padding="same", kernel_initializer=kernel_initializer)
    x = tf.layers.batch_normalization(x, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=kernel_initializer)
    x = activation(x + inputs)

    return x


def leaky_relu(x, alpha):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def encoder(inputs):
    x = inputs
    h = 128
    latent_dim = 128
    initializer = tf.random_normal_initializer(0, 0.02)

    # -------
    x = tf.layers.conv2d(x, h, kernel_size=5, strides=(1, 1), padding="same", kernel_initializer=initializer)
    x = tf.layers.average_pooling2d(x, 2, 2, padding='same')
    x = tf.nn.relu(x)

    # -------
    x = res_block(x, h, kernel_size=3, kernel_initializer=initializer)
    x = tf.layers.average_pooling2d(x, 2, 2, padding='same')

    # -------
    x = res_block(x, h, kernel_size=3, kernel_initializer=initializer)
    x = tf.layers.average_pooling2d(x, 2, 2, padding='same')

    # -------
    x = res_block(x, h, kernel_size=3, kernel_initializer=initializer)

    # -------
    x = tf.reshape(x, shape=(-1, h * 4 * 4))
    x = tf.layers.dense(x, latent_dim, kernel_initializer=initializer)

    return x



def generator(inputs):
    h = 128
    initializer = tf.random_normal_initializer(0, 0.02)

    x = tf.layers.dense(inputs, h * 4 * 4, kernel_initializer=initializer)
    x = tf.reshape(x, shape=(-1, 4, 4, h))
    x = tf.nn.relu(x)

    # -------
    x = res_block(x, h, kernel_size=3, kernel_initializer=initializer)
    x = tf.image.resize_nearest_neighbor(x, (x.shape[1] * 2, x.shape[2] * 2))

    # -------
    x = res_block(x, h, kernel_size=3, kernel_initializer=initializer)
    x = tf.image.resize_nearest_neighbor(x, (x.shape[1] * 2, x.shape[2] * 2))

    # -------
    x = res_block(x, h, kernel_size=3, kernel_initializer=initializer)
    x = tf.image.resize_nearest_neighbor(x, (x.shape[1] * 2, x.shape[2] * 2))

    # -------
    x = res_block(x, h, kernel_size=3, kernel_initializer=initializer)
    x = tf.layers.conv2d(x, 1, kernel_size=1, padding="same", kernel_initializer=initializer)

    x = tf.nn.tanh(x)

    return x



def discriminator(inputs):
    x = inputs
    h = 128
    initializer = tf.random_normal_initializer(0, 0.02)
    lrelu = lambda x: leaky_relu(x, 0.2)

    # -------
    x = tf.layers.conv2d(x, h, kernel_size=5, strides=(1, 1), padding="same", kernel_initializer=initializer)
    x = tf.layers.average_pooling2d(x, 2, 2, padding='same')
    x = lrelu(x)

    # -------
    x = res_block(x, h, kernel_size=3, activation=lrelu, kernel_initializer=initializer)
    x = tf.layers.average_pooling2d(x, 2, 2, padding='same')

    # -------
    x = res_block(x, h, kernel_size=3, activation=lrelu, kernel_initializer=initializer)
    x = tf.layers.average_pooling2d(x, 2, 2, padding='same')

    # -------
    x = res_block(x, h, kernel_size=3, activation=lrelu, kernel_initializer=initializer)

    # -------
    x = tf.reshape(x, shape=(-1, h * 4 * 4))
    x = tf.layers.dense(x, 1, kernel_initializer=initializer)

    x = tf.sigmoid(x)

    return x


def code_discriminator(inputs):
    x = inputs
    h = 700
    latent_dim = 128
    initializer = tf.random_normal_initializer(0, 0.02)
    lrelu = lambda x: leaky_relu(x, 0.2)

    # -------
    x = tf.layers.dense(x, h, kernel_initializer=initializer)
    x = lrelu(x)

    # -------
    x = tf.layers.dense(x, h, kernel_initializer=initializer)
    x = lrelu(x)

    # -------
    x = tf.layers.dense(x, 1, kernel_initializer=initializer)
    x = tf.sigmoid(x)

    return x

