import tensorflow as tf
import numpy as np

def standard_normal_log_likelihood(x):
    log_likelihood = - 0.5 * tf.square(x) - tf.log(2.0 * np.pi)
    return log_likelihood


class CouplingLayer(object):
    def __init__(self, direction, name):
        self.direction = direction
        self.name = name


class RealNVP(object):
    def __init__(self, config):
        """TODO"""
        self.config = config
        self.build()

    def build(self):
        """TODO"""
        self.add_placeholders()
        self.add_layers()
        self.add_encoder_decoder_ops()
        self.add_loss_ops()
        self.add_training_ops()

    def add_placeholders(self):
        """TODO"""
        D_in = self.config.D_in

        self.x_placeholder = tf.placeholder(
            shape=(None, D_in), dtype=tf.float32)

        self.z_placeholder = tf.placeholder(
            shape=(None, D_in), dtype=tf.float32)

    def create_feed_dict(self, x_batch=None, z_batch=None):
        """TODO"""
        feed_dict = {}

        if x_batch is not None:
            feed_dict[self.x_placeholder] = x_batch

        if z_batch is not None:
            feed_dict[self.z_placeholder] = z_batch

        return feed_dict

    def add_forward_preprocessing_ops(self):
        return self.x_placeholder

    def add_forward_postprocessing_ops(self, f_x):
        return f_x

    def add_backward_preprocessing_ops(self):
        return self.z_placeholder

    def add_backward_postprocessing_ops(self, f_z):
        return f_z

    def add_layers(self, params, weight_norm=True, train=True):
        """Create coupling layers"""
        self.layers = []

    def add_encoder_decoder_ops(self):
        """TODO"""
        train = self.config.mode == "train"

        x = self.add_forward_preprocessing_ops()

        forward_out = x
        for layer in self.layers:
            forward_out = layer.forward(forward_out)

        self.z = self.add_forward_postprocessing_ops(forward_out)

        z = self.add_backward_preprocessing_ops()
        backward_out = z
        for layer in reversed(self.layers):
            backward_out = layer.backward(backward_out)

        self.x = self.add_backward_postprocessing_ops(backward_out)

    def add_loss_ops(self):
        """TODO"""
        log_likelihood = tf.reduce_sum(
            standard_normal_log_likelihood(self.z), axis=1)

        log_likelihood_loss = -tf.reduce_mean(
            log_likelihood + self.log_det_jacobian)

        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = tf.sum(reg_variables)

        self.forward_loss = -log_likelihood_loss + reg_loss

    def add_training_ops(self):
        """TODO: regularization? logging? check gradients?"""
        optimizer = tf.train.AdamOptimizer(
            self.config.learning_rate,
            use_locking=False,
            name="Adam",
        )

        self.global_step = tf.get_variable(
            "global_step", (), tf.int64,
            tf.zeros_initializer(), trainable=False)

        self.train_op = optimizer.minimize(loss=self.forward_loss,
                                           global_step=self.global_step)

    def train_encoder_on_batch(self, sess, x_batch, z_batch):
        """TODO"""
        feed_dict = self.create_feed_dict(x_batch=x_batch)
        _, forward_loss = sess.run((self.train_op, self.forward_loss),
                           feed_dict=feed_dict)
        return forward_loss

    def forward_on_batch(self, sess, x_batch):
        """TODO"""
        feed_dict = self.create_feed_dict(x_batch=x_batch)
        z = sess.run(self.z, feed_dict=feed_dict)
        return z

    def train_decoder_on_batch(self, sess, x_batch, z_batch):
        """TODO"""
        raise NotImplementedError(
            "Backwards training not implemented."
            " Define a backward loss and implement this method.")

    def backward_on_batch(self, sess, z_batch):
        feed_dict = self.create_feed_dict(z_batch=z_batch)
        x = sess.run(self.x, feed_dict=feed_dict)
        return x
