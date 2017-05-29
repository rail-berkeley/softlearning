import tensorflow as tf

WEIGHT_DEFAULT_NAME = "weights"
BIAS_DEFAULT_NAME = "bias"


def weight_variable(
        shape,
        initializer=None,
        name=WEIGHT_DEFAULT_NAME,
):
    """
    Return a variable with the given shape.

    :param initializer: TensorFlow initializer. Default Xavier.
    :param name: Variable name.
    :param shape: Variable shape.
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

    :param initializer: TensorFlow initializer. Default zero.
    :param name: Variable name.
    :param shape: Variable shape.
    """
    if initializer is None:
        initializer = tf.constant_initializer(0.)

    return weight_variable(shape,
                           initializer=initializer,
                           name=name)


def batch_matmul(a, b):
    """ Batch matrix multiplication.

    Second argument, 'b', should be a rank-2 tensor. Performs a vector-matrix
    multiplication along the last axis of 'a' for each leading dimension of 'a'.
    Supports maximum of 2 leading axis.

    :param a: Tensor of shape ... x N
    :param b: Rank-2 tensor of shape N x M
    :return: Tensor of shape ... x M
    """
    assert b.get_shape().ndims == 2

    a_n_dims = a.get_shape().ndims
    b_n_dims = b.get_shape().ndims
    assert b_n_dims == 2

    if a_n_dims == 2:
        return tf.matmul(a, b)
    if a_n_dims == 3:
        return tf.einsum('aij,jk->aik', a, b)
    else:
        raise ValueError


def affine(
        inp,
        units,
        bias=True,
        W_initializer=None,
        b_initializer=None,
        W_name=WEIGHT_DEFAULT_NAME,
        bias_name=BIAS_DEFAULT_NAME,
):
    """ Create an affine layer.

    :param inp: Input tensor.
    :param units: Number of units.
    :param bias: Include bias term.
    :param W_initializer: Initializer for the multiplicative weight.
    :param b_initializer: Initializer for the bias term.
    :param W_name: Name of the weight.
    :param bias_name: Name of the bias.
    :return: Tensor defined as input.dot(weight) + bias
    """
    input_size = inp.get_shape()[-1].value
    W = weight_variable([input_size, units],
                        initializer=W_initializer,
                        name=W_name)

    output = batch_matmul(inp, W)

    if bias:
        b = bias_variable((units, ),
                          initializer=b_initializer,
                          name=bias_name)

        output += b

    return output


def mlp(inputs,
        layer_sizes,
        nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.tanh,
        W_initializer=None,
        b_initializer=None):
    """
    Create a multi-layer perceptron with the given hidden sizes. The
    nonlinearity is applied after every hidden layer.

    Supports input tensors of rank 2 and rank 3. All inputs should have the same
    tensor rank. It is assumed that the vectors along the last axis form the
    samples, and the mlp is applied independently to each leading dimension.
    If multiple inputs are provided, then the corresponding rank-1 vectors
    are concatenated. The leading dimensions of the network output are equal
    to the 'outer product' of the inputs' shapes. For example

    input 1 shape: N x K x D1
    input 2 shape: N x 1 x D2

    output shape: N x K x (number of output units))

    :param inputs: List of input tensors.
    :param layer_sizes: List of layers sizes, including the output layer.
    :param nonlinearity: Hidden layer nonlinearity.
    :param output_nonlinearity: Output layer nonlinearity.
    :param W_initializer: Weight initializer.
    :param b_initializer: Bias initializer.
    :return:
    """
    if type(inputs) is tf.Tensor:
        inputs = [inputs]

    # Take care of the input layer separately to make use of broadcasting in
    # the case of several inputs
    with tf.variable_scope('layer0'):
        layer = bias_variable(layer_sizes[0], b_initializer)
        for i, inp in enumerate(inputs):
            with tf.variable_scope('input' + str(i)):
                layer += affine(
                    inp=inp,
                    units=layer_sizes[0],
                    bias=False,
                    W_initializer=W_initializer,
                    b_initializer=b_initializer
                )

        layer = nonlinearity(layer)

    for i_layer, size in enumerate(layer_sizes[1:], 1):
        with tf.variable_scope('layer{0}'.format(i_layer)):
            layer = affine(layer, size,
                           W_initializer=W_initializer,
                           b_initializer=b_initializer)
            if i_layer < len(layer_sizes) - 1:
                layer = nonlinearity(layer)

    if output_nonlinearity is not None:
        layer = output_nonlinearity(layer)

    return layer
