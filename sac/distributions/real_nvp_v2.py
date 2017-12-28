import json
import tensorflow as tf
import numpy as np

EPS = 1e-9

def standard_normal_log_likelihood(x):
    dist = tf.contrib.distributions.MultivariateNormalDiag(
        loc=tf.zeros(x.shape[1:]), scale_diag=tf.ones(x.shape[1:]))
    log_probs = dist.log_prob(x)
    return log_probs

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
        with tf.variable_scope(self.name):
            shape = inputs.get_shape()
            mask = self.get_mask(inputs, dtype=inputs.dtype)

            # masked half of inputs
            masked_inputs = inputs * mask

            # TODO: scale and translation could be merged into a single network
            with tf.variable_scope("scale", reuse=tf.AUTO_REUSE):
                scale = mask * self.scale_fn(masked_inputs)

            with tf.variable_scope("translation", reuse=tf.AUTO_REUSE):
                translation = mask * self.translation_fn(masked_inputs)

            # TODO: check the masks
            exp_scale = tf.check_numerics(
                tf.exp(scale), "tf.exp(scale) contains NaNs or Infs.")
            # (9) in paper

            if self.parity == "odd":
                outputs = tf.stack((
                    inputs[:, 0] * exp_scale[:, 1] + translation[:, 1],
                    inputs[:, 1],
                ), axis=1)
            else:
                outputs = tf.stack((
                    inputs[:, 0],
                    inputs[:, 1] * exp_scale[:, 0] + translation[:, 0],
                ), axis=1)

            log_det_jacobian = tf.reduce_sum(
                scale, axis=tuple(range(1, len(shape))))

            return outputs, log_det_jacobian

    def backward(self, inputs):
        """Calculate inverse of the layer

        Note that `inputs` correspond to the `outputs` in forward function
        """
        with tf.variable_scope(self.name):
            mask = self.get_mask(inputs, dtype=inputs.dtype)

            masked_inputs = inputs * mask

            # TODO: scale and translation could be merged into a single network
            with tf.variable_scope("scale", reuse=tf.AUTO_REUSE):
                scale = mask * self.scale_fn(masked_inputs)

            with tf.variable_scope("translation", reuse=tf.AUTO_REUSE):
                translation = mask * self.translation_fn(masked_inputs)

            if self.parity == "odd":
                outputs = tf.stack((
                    (inputs[:, 0] - translation[:, 1]) * tf.exp(-scale[:, 1]),
                    inputs[:, 1],
                ), axis=1)
            else:
                outputs = tf.stack((
                    inputs[:, 0],
                    (inputs[:, 1] - translation[:, 0]) * tf.exp(-scale[:, 0]),
                ), axis=1)

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

        bias_initializer = tf.initializers.random_normal()
        bias = tf.get_variable(
            name="bias_{i}".format(i=i),
            shape=(layer_size,),
            initializer=bias_initializer)

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
    "num_coupling_layers": 2,
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

        self.x_placeholder = tf.placeholder(
            shape=(None, D_in), dtype=tf.float32, name="x_placeholder")

        self.z_placeholder = tf.placeholder(
            shape=(None, D_in), dtype=tf.float32, name="z_placeholder")

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
                layer_sizes=(
                    *translation_hidden_sizes,
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

        self.layers = [
            CouplingLayer(
                parity=("even", "odd")[i % 2],
                name="coupling_{i}".format(i=i),
                translation_fn=translation_wrapper,
                scale_fn=scale_wrapper
            )
            for i in range(1, num_coupling_layers + 1)
        ]

    def add_encoder_decoder_ops(self):
        """TODO"""
        train = self.config["mode"] == "train"

        # Encoder
        x = self.add_forward_preprocessing_ops()  # (N, D)
        # Following could be tf.zeros((None,)), but zeros does not accept
        # None dimension in the shape. The result of this is just zero
        # tensor with shape (None, ), i.e. (N,), where N is the batch_size
        self.sum_log_det_jacobians = tf.reduce_sum(
            tf.zeros_like(x), axis=tuple(range(1, len(x.shape))))  # (N,)
        forward_out = x
        for layer in self.layers:
            forward_out, log_det_jacobian = layer.forward_and_jacobian(
                forward_out)
            assert (self.sum_log_det_jacobians.shape.as_list()
                    == log_det_jacobian.shape.as_list())

            self.sum_log_det_jacobians += log_det_jacobian  # (N,)

        # self.z = f(x)
        self.z = self.add_forward_postprocessing_ops(forward_out) # (N, D)
        # End Encoder


        # Decoder
        z = self.add_backward_preprocessing_ops()  # (N, D)
        backward_out = z
        for layer in reversed(self.layers):
            backward_out = layer.backward(backward_out)
        self.x = self.add_backward_postprocessing_ops(backward_out)
        # End Decoder

        self.log_p_x = (
            standard_normal_log_likelihood(self.z)
            + self.sum_log_det_jacobians)

    def add_loss_ops(self):
        """TODO"""

        # log_likelihood squeezes by x-axis,
        # log_det_jacobian squeezes by y-axis,
        ll_loss = - tf.reduce_mean(self.log_p_x)

        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = tf.reduce_sum(reg_variables)

        self.loss = ll_loss + reg_loss

    def add_training_ops(self):
        """TODO: regularization? logging? check gradients?"""
        optimizer = tf.train.AdamOptimizer(
            self.config["learning_rate"], use_locking=False)

        self.global_step = tf.get_variable(
            "global_step", (), tf.int64,
            tf.zeros_initializer(), trainable=False)

        self.train_op = optimizer.minimize(loss=self.loss,
                                           global_step=self.global_step)
