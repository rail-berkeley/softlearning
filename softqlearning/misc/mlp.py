import numpy as np
import tensorflow as tf

WEIGHT_DEFAULT_NAME = "weights"
BIAS_DEFAULT_NAME = "bias"


def weight_variable_triu_exp(
        dim,
        initializer,
        name=WEIGHT_DEFAULT_NAME,
        eps=1-5,
):
    """
    Construct lower triangular matrix with exponentiated diagonal
    of dimension DIM.
    """

    if initializer is None:
        initializer = tf.contrib.layers.xavier_initializer()

    inds = np.stack(np.triu_indices(dim, k=1)).T
    n_vars = int((dim + 1) / 2 * dim)
    var = tf.get_variable(name, (n_vars,), initializer=initializer)

    W = tf.sparse_tensor_to_dense(
            tf.SparseTensor(indices=inds, values=var[dim:],
                            dense_shape=[dim, dim])
        ) + tf.diag(tf.exp(var[:dim]) + eps)

    return W


def weight_variable(
        shape,
        initializer,
        name=WEIGHT_DEFAULT_NAME,
):
    """
    Return a variable with the given shape.

    :param initializer: TensorFlow initializer
    :param reuse_variables:
    :param name:
    :param shape:
    """
    if initializer is None:
        initializer = tf.contrib.layers.xavier_initializer()

    var = tf.get_variable(name, shape, initializer=initializer)
    return var


def bias_variable(
        shape,
        initializer=None,
        name=BIAS_DEFAULT_NAME,
):
    """
    Return a bias variable with the given shape.

    :param initializer: TensorFlow initializer
    :param reuse_variables:
    :param name:
    :param shape:
    """
    if initializer is None:
        initializer = tf.constant_initializer(0.)

    return weight_variable(shape,
                           initializer=initializer,
                           name=name)


def batch_matmul(a, b):
    assert b.get_shape().ndims == 2

    a_ndims = a.get_shape().ndims
    b_ndims = b.get_shape().ndims
    assert b_ndims == 2

    if a_ndims == 2:
        return tf.matmul(a, b)
    if a_ndims == 3:
        return tf.einsum('aij,jk->aik', a, b)
    else:
        raise ValueError


def affine(
        input,
        units,
        bias=True,
        W_initializer=None,
        b_initializer=None,
        W_name=WEIGHT_DEFAULT_NAME,
        bias_name=BIAS_DEFAULT_NAME,
        full_rank=False,
):
    """
    Create a linear layer.

    :param W_initializer:
    :param b_initializer:
    :param reuse_variables:
    :param bias_name: String for the bias variables names
    :param W_name: String for the weight matrix variables names
    :param last_layer: Input tensor
    :param last_size: Size of the input tensor
    :param new_size: Size of the output tensor
    :return:
    """
    input_size = input.get_shape()[-1].value
    if full_rank:
        if input_size == units:
            W = weight_variable_triu_exp(units, W_initializer, W_name)
        else:
            # First take random projection into output space
            proj = np.random.randn(input_size, units) / np.sqrt(input_size)
            proj = tf.constant(proj, dtype=tf.float32)

            # Then multiply by an invertible weight matrix
            triu = weight_variable_triu_exp(units, W_initializer, W_name)
            # import ipdb; ipdb.set_trace()
            W = tf.matmul(proj, triu)
    else:
        W = weight_variable([input_size, units],
                            initializer=W_initializer,
                            name=W_name)

    output = batch_matmul(input, W)

    if bias:
        b = bias_variable((units, ),
                          initializer=b_initializer,
                          name=bias_name)

        output += tf.expand_dims(b, 0)

    return output


def mlp(inputs,
        layer_sizes,
        nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.tanh,
        W_initializer=None,
        b_initializer=None,
        full_rank=False):
    """
    Create a multi-layer perceptron with the given hidden sizes. The
    nonlinearity is applied after every hidden layer.

    :param b_initializer:
    :param W_initializer:
    :param reuse_variables:
    :param input: tf.Tensor, or placeholder, input to mlp
    :param hidden_sizes: int iterable of the hidden sizes
    :param nonlinearity: the initialization function for the nonlinearity
    :return: Output of MLP.
    :type: tf.Tensor
    """
    # TODO: test if works as expected

    # TODO: comment: if given list of inputs, performs fancy broadcasting thing.
    if type(inputs) is tf.Tensor:
        inputs = [inputs]

    # Take care of the input layer separately to make use of broadcasting in
    # the case of several inputs
    with tf.variable_scope('layer0'):
        layer = bias_variable(layer_sizes[0], b_initializer)
        for i, input in enumerate(inputs):
            with tf.variable_scope('input' + str(i)):
                layer += affine(input, layer_sizes[0],
                                W_initializer, b_initializer, False,
                                full_rank=full_rank)

        layer = nonlinearity(layer)

    for i_layer, size in enumerate(layer_sizes[1:]):
        with tf.variable_scope('layer{0}'.format(i_layer + 1)):
            layer = affine(layer,
                           size,
                           W_initializer=W_initializer,
                           b_initializer=b_initializer,
                           full_rank=full_rank)
            if i_layer < len(layer_sizes) - 2:
                layer = nonlinearity(layer)

    if output_nonlinearity is not None:
        layer = output_nonlinearity(layer)

    return layer
