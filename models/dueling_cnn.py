import tensorflow as tf
from .base import BaseNet


class Net(BaseNet):

    def __init__(self, n_class, weight_decay=0.00001, name='simple_cnn'):
        super().__init__(name)

        self.weight_decay = weight_decay
        self.data_format = 'channels_first'
        self.n_class = n_class

    def call(self, images, is_training):
        w_args = {
            'kernel_initializer': tf.initializers.variance_scaling(1),
            'kernel_regularizer': lambda w: self.weight_decay * tf.nn.l2_loss(w)
        }
        conv_args = {
            'data_format': self.data_format,
            'padding': 'same',
            **w_args
        }
        fc_args = {
            **w_args
        }

        net = tf.layers.conv2d(images, 32, 8, 4, name='conv1', **conv_args)
        net = tf.nn.relu(net)
        net = tf.layers.conv2d(net, 64, 4, 2, name='conv2', **conv_args)
        net = tf.nn.relu(net)
        net = tf.layers.conv2d(net, 64, 3, 1, name='conv3', **conv_args)
        net = tf.nn.relu(net)

        net = tf.layers.flatten(net)

        # advantage function
        anet = tf.layers.dense(net, 512, name='fc1', **fc_args)
        anet = tf.nn.relu(anet)
        anet = tf.layers.dense(anet, self.n_class, name='fc2', **fc_args)

        # state value function
        snet = tf.layers.dense(net, 512, name='fc3', **fc_args)
        snet = tf.nn.relu(snet)
        snet = tf.layers.dense(snet, 1, name='fc4', **fc_args)

        # Q function
        net = snet + (anet - tf.reduce_mean(anet, axis=1, keepdims=True))
        return net
