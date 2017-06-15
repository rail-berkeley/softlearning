import math
from collections import namedtuple

import numpy as np
import tensorflow as tf

from railrl.pythonplusplus import identity

REGULARIZABLE_VARS = "regularizable_weights_collection"
WEIGHT_DEFAULT_NAME = "weights"
BIAS_DEFAULT_NAME = "bias"
BN_SCALE_DEFAULT_NAME = "bn_scale"
BN_OFFSET_DEFAULT_NAME = "bn_offset"
BN_POP_MEAN_DEFAULT_NAME = "bn_pop_means"
BN_POP_VAR_DEAFULT_NAME = "bn_pop_var"
LAYER_NORM_BIAS_DEFAULT_NAME = "ln_bias"
LAYER_NORM_GAIN_DEFAULT_NAME = "ln_gain"
LAYER_NORMALIZATION_DEFAULT_NAME = "layer_normalization"
_BATCH_NORM_UPDATE_POP_STATS_COLLECTION_ = "_batch_norm_update_pop_stats_"
_UNTRAINABLE_BATCH_NORM_VARS_ = "_untrainable_batch_norm_vars_"

# TODO(vpong): Use this namedtuple when possible
MlpConfig = namedtuple('MlpConfig', ['W_init', 'b_init', 'nonlinearity'])


class BatchNormConfig(object):
    def __init__(
            self,
            enable_scale=True,
            enable_offset=True,
            mean_init=0.,
            std_init=1.,
            decay=0.999,
            epsilon=1e-5,
            bn_scale_name=BN_SCALE_DEFAULT_NAME,
            bn_offset_name=BN_OFFSET_DEFAULT_NAME,
            bn_pop_mean_name=BN_POP_MEAN_DEFAULT_NAME,
            bn_pop_var_name=BN_POP_VAR_DEAFULT_NAME,
    ):
        self.enable_scale = enable_scale
        self.enable_offset = enable_offset
        self.mean_init = mean_init
        self.std_init = std_init
        self.decay = decay
        self.epsilon = epsilon
        self.bn_scale_name = bn_scale_name
        self.bn_offset_name = bn_offset_name
        self.bn_pop_mean_name = bn_pop_mean_name
        self.bn_pop_var_name = bn_pop_var_name


class BatchNormOps(object):
    def __init__(
            self,
            scale,
            offset,
            pop_mean,
            pop_var,
            batch_mean=None,
            batch_var=None,
            update_pop_mean_op=None,
            update_pop_var_op=None,
    ):
        self.scale = scale
        self.offset = offset
        self.pop_mean = pop_mean
        self.pop_var = pop_var
        self.batch_mean = batch_mean
        self.batch_var = batch_var
        assert (
            update_pop_var_op is not None and update_pop_mean_op is not None or
            update_pop_var_op is None and update_pop_mean_op is None
        )
        self.update_pop_mean_op = update_pop_mean_op
        self.update_pop_var_op = update_pop_var_op

        self._update_pop_stats_ops = []
        if self.update_pop_mean_op is not None:
            self._update_pop_stats_ops += [
                self.update_pop_mean_op, self.update_pop_var_op
            ]

    @property
    def update_pop_stats_ops(self):
        return self._update_pop_stats_ops

    def add_pop_stats_ops(self, pop_stat_ops):
        self._update_pop_stats_ops += pop_stat_ops


def get_regularizable_variables(scope):
    """
    Get *all* regularizable variables in the scope.
    :param scope: scope to filter variables by
    :return:
    """
    return tf.get_collection(REGULARIZABLE_VARS, scope)


def add_to_collection_if_not_added(key, value):
    if value in tf.get_collection(key):
        return
    tf.add_to_collection(key, value)


def weight_variable(
        shape,
        initializer=None,
        name=WEIGHT_DEFAULT_NAME,
        regularizable=True,
):
    """
    Return a variable with the given shape.

    :param initializer: TensorFlow initializer
    :param name:
    :param shape:
    """
    if initializer is None:
        initializer = tf.random_uniform_initializer(minval=-3e-3,
                                                    maxval=3e-3)
    var = tf.get_variable(name, shape, initializer=initializer)
    if regularizable:
        add_to_collection_if_not_added(REGULARIZABLE_VARS, var)
    return var


def bias_variable(
        shape,
        initializer=None,
        name=BIAS_DEFAULT_NAME,
        regularizable=False,
):
    """
    Return a bias variable with the given shape.

    :param initializer: TensorFlow initializer
    :param name:
    :param shape:
    """
    if initializer is None:
        initializer = tf.constant_initializer(0.)
    return weight_variable(shape,
                           initializer=initializer,
                           name=name,
                           regularizable=regularizable)


def layer_normalize(
        input_pre_nonlinear_activations,
        input_shape,
        epsilon=1e-5,
        name=LAYER_NORMALIZATION_DEFAULT_NAME,
):
    """
    Layer normalizes a 2D tensor along its second axis, which corresponds to
    normalizing within a layer.

    :param input_pre_nonlinear_activations:
    :param input_shape:
    :param name: Name for the variables in this layer.
    :param epsilon: The actual normalized value is
    ```
        norm = (x - mean) / sqrt(variance + epsilon)
    ```
    for numerical stability.
    :return: Layer-normalized pre-non-linear activations
    """
    mean, variance = tf.nn.moments(input_pre_nonlinear_activations, [1],
                                   keep_dims=True)
    normalised_input = (input_pre_nonlinear_activations - mean) / tf.sqrt(
        variance + epsilon)
    with tf.variable_scope(name):
        gains = tf.get_variable(
            LAYER_NORM_GAIN_DEFAULT_NAME,
            input_shape,
            initializer=tf.constant_initializer(0.),
        )
        biases = tf.get_variable(
            LAYER_NORM_BIAS_DEFAULT_NAME,
            input_shape,
            initializer=tf.constant_initializer(0.),
        )
    return normalised_input * gains + biases


def _get_collection(key, scope):
    if isinstance(scope, tf.VariableScope):
        scope = scope.name
    return tf.get_collection(key, scope=scope)


def get_batch_norm_update_pop_stats_ops(scope=None):
    """
    :param scope: If None, return all ops.
    :return: List of batch norm ops that update population statistics in a
    given scope.
    """
    return _get_collection(_BATCH_NORM_UPDATE_POP_STATS_COLLECTION_, scope)


def get_untrainable_batch_norm_vars(scope=None):
    """
    :param scope: If None, return all ops.
    :return: List of untrainable batch norm variables (i.e. population mean
    and variance) given scope.
    """
    return _get_collection(_UNTRAINABLE_BATCH_NORM_VARS_, scope)


def batch_norm(
        input_tensor,
        is_training,
        batch_norm_config=None,
):
    """
    Based on http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
    Note that unlike the version given above, the caller must explicitly update
    the population mean and variance.

    This can be done by doing:

    ```
    sess = ...
    training_feed_dict = ...
    train_output, batch_norm_op = tf_util.batch_norm(input_tensor, True)
    pop_update_ops = batch_norm_op.update_pop_stats_ops  # returns a list

    output, *_ = sess.run(
        [train_output] + pop_update_ops,
        feed_dict=training_feed_dict
    )
    ```

    If batch_norm is called inside of a scope, `pop_update_ops` can
    alternatively be accessed by running
    ```
    pop_update_ops = tf_util.get_batchnorm_ops(scope)
    ```

    :param input_tensor: Input tensor
    :param is_training: Is this layer in training mode?
    :param batch_norm_config: BatchNormConfig
    :return: tuple, (output Tensor, BatchNormOps)
    """
    if batch_norm_config is None:
        batch_norm_config = BatchNormConfig()  # Use default settings
    decay = batch_norm_config.decay
    epsilon = batch_norm_config.epsilon

    bn_shape = input_tensor.get_shape()[-1]
    scale, offset = None, None
    if batch_norm_config.enable_scale:
        scale = tf.get_variable(
            batch_norm_config.bn_scale_name,
            shape=bn_shape,
            initializer=tf.constant_initializer(1.),
        )
    if batch_norm_config.enable_offset:
        offset = tf.get_variable(
            batch_norm_config.bn_offset_name,
            shape=bn_shape,
            initializer=tf.constant_initializer(0.),
        )
    pop_mean = tf.get_variable(
        batch_norm_config.bn_pop_mean_name,
        shape=bn_shape,
        initializer=tf.constant_initializer(0.),
        trainable=False,
    )
    pop_var = tf.get_variable(
        batch_norm_config.bn_pop_var_name,
        shape=bn_shape,
        initializer=tf.constant_initializer(1.),
        trainable=False,
    )
    add_to_collection_if_not_added(_UNTRAINABLE_BATCH_NORM_VARS_, pop_mean)
    add_to_collection_if_not_added(_UNTRAINABLE_BATCH_NORM_VARS_, pop_var)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(input_tensor, [0])
        update_pop_mean_op = tf.assign(
            pop_mean,
            pop_mean * decay + batch_mean * (1 - decay)
        )
        update_pop_var_op = tf.assign(
            pop_var,
            pop_var * decay + batch_var * (1 - decay)
        )
        add_to_collection_if_not_added(
            _BATCH_NORM_UPDATE_POP_STATS_COLLECTION_,
            update_pop_mean_op
        )
        add_to_collection_if_not_added(
            _BATCH_NORM_UPDATE_POP_STATS_COLLECTION_,
            update_pop_var_op
        )
        return tf.nn.batch_normalization(
            input_tensor, batch_mean, batch_var, offset, scale, epsilon
        ), BatchNormOps(
            scale,
            offset,
            pop_mean,
            pop_var,
            batch_mean=batch_mean,
            batch_var=batch_var,
            update_pop_mean_op=update_pop_mean_op,
            update_pop_var_op=update_pop_var_op,
        )
    else:
        return tf.nn.batch_normalization(
            input_tensor, pop_mean, pop_var, offset, scale, epsilon
        ), BatchNormOps(
            scale,
            offset,
            pop_mean,
            pop_var,
        )


def linear(
        last_layer,
        last_size,
        new_size,
        W_initializer=None,
        b_initializer=None,
        W_name=WEIGHT_DEFAULT_NAME,
        b_name=BIAS_DEFAULT_NAME,
):
    """
    Create a linear layer.

    :param W_initializer:
    :param b_initializer:
    :param b_name: String for the bias variables names
    :param W_name: String for the weight matrix variables names
    :param last_layer: Input tensor
    :param last_size: Size of the input tensor
    :param new_size: Size of the output tensor
    :return:
    """
    W = weight_variable([last_size, new_size],
                        initializer=W_initializer,
                        name=W_name)
    b = bias_variable((new_size,),
                      initializer=b_initializer,
                      name=b_name)
    return tf.matmul(last_layer, W) + tf.expand_dims(b, 0)


def mlp(input_layer,
        input_layer_size,
        hidden_sizes,
        nonlinearity,
        W_initializer=None,
        b_initializer=None,
        pre_nonlin_lambda=identity,
        post_nonlin_lambda=identity,
        ):
    """
    Create a multi-layer perceptron with the given hidden sizes. The
    nonlinearity is applied after every hidden layer.

    :param b_initializer:
    :param W_initializer:
    :param input_layer: tf.Tensor, input to mlp
    :param input_layer_size: int, size of the input
    :param hidden_sizes: int iterable of the hidden sizes
    :param nonlinearity: the initialization function for the nonlinearity
    :param post_nonlin_lambda: A function to pass the post-non-linearity
    values through.
    This is only applied between layers. Not on the input nor the output.
    :param pre_nonlin_lambda: A function to pass the pre-non-linearity
    values through.
    This is only applied between layers. Not on the input nor the output.
    :return: Output of MLP.
    :type: tf.Tensor
    """
    last_layer = input_layer
    last_layer_size = input_layer_size
    for layer, hidden_size in enumerate(hidden_sizes):
        with tf.variable_scope('hidden{0}'.format(layer)) as _:
            pre_nonlin = linear(last_layer,
                                last_layer_size,
                                hidden_size,
                                W_initializer=W_initializer,
                                b_initializer=b_initializer,
                                )
            if layer == len(hidden_sizes) - 1:  # Last layer
                last_layer = nonlinearity(pre_nonlin)
            else:
                last_layer = post_nonlin_lambda(
                    nonlinearity(
                        pre_nonlin_lambda(
                            pre_nonlin
                        )
                    )
                )
            last_layer_size = hidden_size
    return last_layer


def get_lower_triangle_flat_indices(dim):
    indices = []
    for row in range(dim):
        for col in range(dim):
            if col <= row:
                indices.append(row * dim + col)
    return indices


def get_num_elems_in_lower_triangle_matrix(dim):
    return int(dim * (dim + 1) / 2)


# From https://github.com/locuslab/icnn/blob/master/RL/src/naf_nets_dm.py
def vec2lower_triangle(vec, dim):
    """
    Convert a vector M of size (n * m) into a matrix of shape (n, m)
    [[e^M[0],    0,           0,             ...,    0]
     [M[n-1],    e^M[n],      0,      0,     ...,    0]
     [M[2n-1],   M[2n],       e^M[2n+1], 0,  ...,    0]
     ...
     [M[m(n-1)], M[m(n-1)+1], ...,       M[mn-2], e^M[mn-1]]
    """
    L = tf.reshape(vec, [-1, dim, dim])
    if int(tf.__version__.split('.')[1]) >= 10:
        L = tf.matrix_band_part(L, -1, 0) - tf.matrix_diag(
            tf.matrix_diag_part(L)) + tf.matrix_diag(
            tf.exp(tf.matrix_diag_part(L)))
    else:
        L = tf.batch_matrix_band_part(L, -1, 0) - tf.batch_matrix_diag(
            tf.batch_matrix_diag_part(L)) + tf.batch_matrix_diag(
            tf.exp(tf.batch_matrix_diag_part(L)))
    return L


def quadratic_multiply(x, A):
    """
    Compute x^T A x
    :param x: [n x m] matrix
    :param A: [n x n] matrix
    :return: x^T A x
    """
    return tf.matmul(
        x,
        tf.matmul(
            A,
            x
        ),
        transpose_a=True,
    )


def he_uniform_initializer():
    """He weight initialization.

    Weights are initialized with a standard deviation of
    :math:`\\sigma = gain \\sqrt{\\frac{1}{fan_{in}}}` [1]_.

    References
    ----------
    .. [1] Kaiming He et al. (2015):
           Delving deep into rectifiers: Surpassing human-level performance on
           imagenet classification. arXiv preprint arXiv:1502.01852.
    """

    def _initializer(shape, **kwargs):
        if len(shape) == 2:
            fan_in = shape[0]
        elif len(shape) > 2:
            fan_in = np.prod(shape[1:])
        else:
            raise Exception("Shape must be have dimension at least 2.")
        delta = np.sqrt(1.0 / fan_in)
        # TODO(vpong): refactor this common piece of code (e.g. move this to a
        # decorator)
        # tf.get_variable puts "partition_info" as another kwargs, which is
        # unfortunately not supported by tf.random_uniform
        acceptable_keys = ["seed", "name"]
        acceptable_kwargs = {
            key: kwargs[key]
            for key in kwargs
            if key in acceptable_keys
            }
        return tf.random_uniform(shape, minval=-delta, maxval=delta,
                                 **acceptable_kwargs)

    return _initializer


def xavier_uniform_initializer():
    def _initializer(shape, **kwargs):
        if len(shape) == 2:
            n_inputs, n_outputs = shape
        else:
            receptive_field_size = np.prod(shape[:2])
            n_inputs = shape[-2] * receptive_field_size
            n_outputs = shape[-1] * receptive_field_size
        init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
        acceptable_keys = ["seed", "name"]
        acceptable_kwargs = {
            key: kwargs[key]
            for key in kwargs
            if key in acceptable_keys
            }
        return tf.random_uniform(shape, minval=-init_range, maxval=init_range,
                                 **acceptable_kwargs)

    return _initializer
