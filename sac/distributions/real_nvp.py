import json
import tensorflow as tf
import numpy as np

def standard_normal_log_likelihood(x):
    # TODO: the last term should probably have K as coefficient?
    # log_likelihood = - 0.5 * (tf.square(x) + tf.log(2.0 * np.pi))
    log_likelihoods = - 0.5 * tf.square(x) - tf.log(2.0 * np.pi)
    log_likelihood = tf.reduce_sum(log_likelihoods, axis=1)
    return log_likelihood

def checkerboard(shape, parity="even", dtype=tf.bool):
    """TODO: Check this implementation"""
    unit = (
        tf.constant((True, False))
        if parity == "even"
        else tf.constant((False, True)))

    checkerboard = tf.tile(unit, (np.prod(shape) // 2,))
    return tf.cast(tf.reshape(checkerboard, shape), dtype)

class CouplingLayer(object):
    def __init__(self, parity, name, translation_fn, scale_fn):
        self.parity = parity
        self.name = name
        self.translation_fn = translation_fn
        self.scale_fn = scale_fn

    def get_mask(self, x, dtype):
        shape = x.get_shape()
        mask = checkerboard(shape[1:], parity=self.parity, dtype=dtype)

        # TODO: remove assert
        assert mask.get_shape() == shape[1:]

        return mask

    def forward_and_jacobian(self, inputs):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            shape = inputs.get_shape()
            mask = self.get_mask(inputs, dtype=inputs.dtype)

            # masked half of inputs
            masked_inputs = inputs * mask

            # TODO: scale and translation could be merged into a single network
            with tf.variable_scope("scale"):
                scale = self.scale_fn(masked_inputs)

            with tf.variable_scope("translation"):
                translation = self.translation_fn(masked_inputs)

            # TODO: check the masks
            exp_scale = tf.check_numerics(
                tf.exp(scale), "tf.exp(scale) contains NaNs or Infs.")
            # (9) in paper
            outputs = (
                masked_inputs
                + (1.0 - mask) * (inputs * exp_scale + translation))

            log_det_jacobian = tf.reduce_sum(
                scale, axis=tuple(range(1, len(shape))))

            return outputs, log_det_jacobian

    def backward(self, inputs):
        """Calculate inverse of the layer

        Note that `inputs` correspond to the `outputs` in forward function
        """
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            mask = self.get_mask(inputs, dtype=inputs.dtype)

            masked_inputs = inputs * mask

            # TODO: scale and translation could be merged into a single network
            with tf.variable_scope("scale"):
                scale = self.scale_fn(masked_inputs)

            with tf.variable_scope("translation"):
                translation = self.translation_fn(masked_inputs)

            outputs = (
                masked_inputs
                + (inputs * (1.0 - mask) - translation) * tf.exp(-scale))

            return outputs

def feedforward_net(inputs,
                    layer_sizes,
                    activation_fn=tf.nn.tanh,
                    output_nonlinearity=None,
                    regularizer=None):
    prev_size = inputs.get_shape().as_list()[-1]
    out = inputs
    for i, layer_size in enumerate(layer_sizes):
        weight_initializer = tf.contrib.layers.xavier_initializer()
        weight = tf.get_variable(
            name="weight_{i}".format(i=i),
            shape=(prev_size, layer_size),
            initializer=weight_initializer,
            regularizer=regularizer)
        bias = tf.get_variable(
            name="bias_{i}".format(i=i),
            initializer=tf.random_normal([layer_size])),
        prev_size = layer_size
        z = tf.matmul(out, weight) + bias

        if i < len(layer_sizes) - 1:
            out = activation_fn(z)
        elif output_nonlinearity is not None:
            out = output_nonlinearity(z)
        else:
            out = z

    return out


DEFAULT_CONFIG = {
    "mode": "train",
    "D_in": 2,
    "learning_rate": 5e-4,
    "scale_regularization": 5e2,
    "num_coupling_layers": 10,
    "translation_hidden_sizes": (25,),
    "scale_hidden_sizes": (25,),
    "squash": False
}

class RealNVP(object):
    def __init__(self, config=None):
        """TODO"""
        self.config = dict(DEFAULT_CONFIG, **(config or {}))
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
        D_in = self.config["D_in"]

        self.batch_size = tf.placeholder_with_default(
            64, (), name="batch_size"
        )

        x_default = tf.random_normal((self.batch_size, D_in))
        self.x_placeholder = tf.placeholder_with_default(
            x_default, (None, D_in), name="x_placeholder")

        self.z_placeholder = tf.placeholder(
            shape=(None, D_in), dtype=tf.float32)

        self.Q_placeholder = tf.placeholder(
            shape=(None,), dtype=tf.float32)

    def create_feed_dict(self,
                         x_batch=None,
                         z_batch=None,
                         Q_batch=None,
                         batch_size=None):
        """TODO"""
        feed_dict = {}

        if x_batch is not None:
            feed_dict[self.x_placeholder] = x_batch

        if z_batch is not None:
            feed_dict[self.z_placeholder] = z_batch

        if Q_batch is not None:
            feed_dict[self.Q_placeholder] = Q_batch

        if batch_size is not None:
            feed_dict[self.batch_size] = batch_size

        return feed_dict

    def add_forward_preprocessing_ops(self):
        return self.x_placeholder

    def add_forward_postprocessing_ops(self, f_x):
        return f_x

    def add_backward_preprocessing_ops(self):
        return self.z_placeholder

    def add_backward_postprocessing_ops(self, f_z):
        return f_z

    def add_layers(self):
        """Create coupling layers"""
        num_coupling_layers = self.config["num_coupling_layers"]
        translation_hidden_sizes = self.config["translation_hidden_sizes"]
        scale_hidden_sizes = self.config["scale_hidden_sizes"]

        def translation_wrapper(inputs):
            return feedforward_net(
                inputs,
                layer_sizes=(*translation_hidden_sizes,
                             self.x_placeholder.shape.as_list()[-1]))

        def scale_wrapper(inputs):
            return feedforward_net(
                inputs,
                layer_sizes=(
                    *scale_hidden_sizes,
                    self.x_placeholder.shape.as_list()[-1]),
                regularizer=tf.contrib.layers.l2_regularizer(
                    self.config["scale_regularization"])
            )

        # parity, name, translation_fn, scale_fn
        self.layers = [
            CouplingLayer(
                parity=("even", "odd")[i % 2 == 0],
                name="coupling_{i}".format(i=i),
                translation_fn=translation_wrapper,
                scale_fn=scale_wrapper
            )
            for i in range(1, num_coupling_layers + 1)
        ]

    def add_encoder_decoder_ops(self):
        """TODO"""
        train = self.config["mode"] == "train"

        x = self.add_forward_preprocessing_ops()  # (N, D)

        self.sum_log_det_jacobians = np.zeros_like(x)  # (N, D)

        forward_out = x
        for layer in self.layers:
            forward_out, log_det_jacobian = layer.forward_and_jacobian(
                forward_out)
            self.sum_log_det_jacobians += log_det_jacobian  # (N, D)

        # self.z = f(x)
        self.z = self.add_forward_postprocessing_ops(forward_out)
        # self.log_p_z = log (p_{Z}(f(x)))
        self.log_p_z = (
            standard_normal_log_likelihood(x)
            + self.sum_log_det_jacobians)  # (N,)

        z = self.add_backward_preprocessing_ops()
        backward_out = z
        for layer in reversed(self.layers):
            backward_out = layer.backward(backward_out)

        self.x = self.add_backward_postprocessing_ops(backward_out)

    def add_loss_ops(self):
        """TODO"""

        reinforce_loss = tf.reduce_mean(
            self.log_p_z * tf.stop_gradient(self.log_p_z - self.Q_placeholder))

        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = tf.reduce_sum(reg_variables)

        self.forward_loss = reinforce_loss + reg_loss

    def add_training_ops(self):
        """TODO: regularization? logging? check gradients?"""
        optimizer = tf.train.AdamOptimizer(
            self.config["learning_rate"],
            use_locking=False,
            name="Adam",
        )

        self.global_step = tf.get_variable(
            "global_step", (), tf.int64,
            tf.zeros_initializer(), trainable=False)

        self.train_op = optimizer.minimize(loss=self.forward_loss,
                                           global_step=self.global_step)

    def train_on_batch(self, session, z_batch, Q_batch):
        """TODO"""
        feed_dict = self.create_feed_dict(z_batch=z_batch, Q_batch=Q_batch)
        _, forward_loss = session.run((self.train_op, self.forward_loss),
                                      feed_dict=feed_dict)
        return forward_loss

    def forward_on_batch(self, session, x_batch=None, batch_size=None):
        """TODO"""
        feed_dict = self.create_feed_dict(
            x_batch=x_batch, batch_size=batch_size)
        z = session.run(self.z, feed_dict=feed_dict)
        return z

    def sample_z(self, session, N=32, return_x=True):
        feed_dict = self.create_feed_dict(batch_size=N)
        fetches = (self.x_placeholder, self.z) if return_x else self.z
        return session.run(fetches, feed_dict=feed_dict)

    def train_decoder_on_batch(self, session, x_batch, z_batch):
        """TODO"""
        raise NotImplementedError(
            "Backwards training not implemented."
            " Define a backward loss and implement this method.")

    def backward_on_batch(self, session, z_batch):
        feed_dict = self.create_feed_dict(z_batch=z_batch)
        x = session.run(self.x, feed_dict=feed_dict)
        return x
