import tensorflow as tf
import numpy as np

def standard_normal_log_likelihood(x):
    # TODO: the last term should probably have K as coefficient?
    # log_likelihood = - 0.5 * (tf.square(x) + tf.log(2.0 * np.pi))
    log_likelihood = - 0.5 * tf.square(x) - tf.log(2.0 * np.pi)
    return log_likelihood

def checkerboard(shape, parity="even", dtype=tf.bool):
    """TODO: Check this implementation"""
    unit = (
        tf.constant((True, False))
        if parity == "even"
        else  tf.constant((False, True)))

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

class Config(object):
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

    def toJSON(self, separators=(',', ': ')):
        return json.dumps(self.__dict__, sort_keys=True,
                          indent=2, separators=separators)

    def __str__(self):
        return "{}({})".format(self.__class__.__name__,
                               self.toJSON())
    def __repr__(self):
        return str(self)

def feedforward_net(inputs,
                    layer_sizes=(10,),
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

DEFAULT_CONFIG = Config(
    mode="train",
    D_in=2,
    scale_regularization=5e2,
    residual_blocks=2,
    n_couplings=2,
    n_scale=4,
    learning_rate=8e-3,
    momentum=1e-1,
    decay=1e-3,
    l2_coeff=5e-5,
    clip_gradient=100.0,
    optimizer="adam",
    dropout_mask=0,
    base_dim=32,
    bottleneck=0,
    use_batch_norm=1,
    alternate=1,
    use_aff=1,
    skip=1,
    data_constraint=0.9,
    n_opt=0)

class RealNVP(object):
    def __init__(self, config=DEFAULT_CONFIG):
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

    def add_layers(self):
        """Create coupling layers"""
        num_coupling_layers = 10

        def translation_wrapper(inputs):
            return feedforward_net(
                inputs,
                layer_sizes=(10, self.x_placeholder.shape.as_list()[-1]))

        def scale_wrapper(inputs):
            return feedforward_net(
                inputs,
                layer_sizes=(10, self.x_placeholder.shape.as_list()[-1]),
                regularizer=tf.contrib.layers.l2_regularizer(
                    self.config.scale_regularization)
            )

        # parity, name, translation_fn, scale_fn
        self.layers = [
            CouplingLayer(
                parity="even" if i % 2 == 0 else "odd",
                name="coupling_{i}".format(i=i),
                translation_fn=translation_wrapper,
                scale_fn=scale_wrapper
            )
            for i in range(1, num_coupling_layers+1)
        ]

    def add_encoder_decoder_ops(self):
        """TODO"""
        train = self.config.mode == "train"

        x = self.add_forward_preprocessing_ops()

        self.sum_log_det_jacobians = np.zeros_like(x)

        forward_out = x
        for layer in self.layers:
            forward_out, log_det_jacobian = layer.forward_and_jacobian(
                forward_out)
            self.sum_log_det_jacobians += log_det_jacobian

        self.z = self.add_forward_postprocessing_ops(forward_out)

        z = self.add_backward_preprocessing_ops()
        backward_out = z
        for layer in reversed(self.layers):
            backward_out = layer.backward(backward_out)

        self.x = self.add_backward_postprocessing_ops(backward_out)

    def add_loss_ops(self):
        """TODO"""
        self.log_likelihood = tf.reduce_sum(
            standard_normal_log_likelihood(self.z),
            axis=tuple(range(1, len(self.z.shape))))

        log_likelihood_loss = -tf.reduce_mean(
            self.log_likelihood + self.sum_log_det_jacobians)

        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = tf.reduce_sum(reg_variables)

        self.forward_loss = log_likelihood_loss + reg_loss

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
