import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable

from softqlearning.misc.mlp import mlp
from softqlearning.misc.tf_proxy import SerializableTensor


class InputBounds(SerializableTensor):
    """
    Modifies the gradient of a given graph ('output') with respect to its
    input so that the gradient always points towards the inputs domain.
    It is assumed that the input domain is L_\infty unit ball.

    'InputBounds' can be used to implement the SVGD algorithm, which assumes a
    target distribution with infinite action support: 'InputBounds' allows
    actions to temporally violate the boundaries, but the modified gradient will
    eventually bring them back within boundaries.
    """
    SLOPE = 10  # This is the new gradient outside the input domain.

    def __init__(self, inp, output):
        """
        :param inp: Input tensor with a constrained domain.
        :param output: Output tensor, whose gradient will be modified.
        """
        Serializable.quick_init(self, locals())

        violation = tf.maximum(tf.abs(inp) - 1, 0)
        total_violation = tf.reduce_sum(violation, axis=-1, keep_dims=True)

        # Expand the first dimension to match the graph
        # (needed for tf.where which does not support broadcasting).
        expanded_total_violation = total_violation * tf.ones_like(output)

        bounded_output = tf.where(tf.greater(expanded_total_violation, 0),
                                  - self.SLOPE * expanded_total_violation,
                                  output)
        super(InputBounds, self).__init__(bounded_output)


class NeuralNetwork(SerializableTensor):
    """ Multilayer Perceptron that support broadcasting.

    See documentation of 'mlp' for information in regards to broadcasting.
    """

    def __init__(self, layer_sizes, inputs, n_heads=1,
                 nonlinearity=tf.nn.relu, output_nonlinearity=None):
        """
        :param layer_sizes: List of # of units at each layer, including output
           layer.
        :param inputs: List of input tensors. See note of broadcasting.
        :param n_heads: Number of heads. The output shape is
            (leading dimensions) x n_heads x (number of output units). If
            n_heads == 1, then the corresponding dimensions is dropped.
        :param nonlinearity: Nonlinearity operation for hidden layers.
        :param output_nonlinearity: Nonlinearity operation for output layer.
        """
        SerializableTensor.quick_init(self, locals())

        if n_heads > 1:
            n_outputs = layer_sizes[-1]
            layer_sizes[-1] *= n_heads

        graph = mlp(inputs,
                    layer_sizes,
                    nonlinearity=nonlinearity,
                    output_nonlinearity=output_nonlinearity)

        if n_heads > 1:
            leading_shape = tf.shape(graph)[:-1]
            new_shape = tf.concat((leading_shape, (n_heads, n_outputs)), 0)
            graph = tf.reshape(graph, new_shape)

        super(NeuralNetwork, self).__init__(graph)


class StochasticNeuralNetwork(NeuralNetwork):
    """
    StochasticNeuralNetwork is like NeuralNetwork, but it feeds an additional
    random vector to the network. The shape of this vector is

    ... x K x (number of output units)

    where ... denotes all leading dimensions of the inputs (all but the last
    axis), and K is the number of outputs samples produced per data point.

    Example:

    input 1 shape: ... x D1
    input 2 shape: ... x D2
    output shape: ... x K x (output shape)

    If K == 1, the corresponding axis is dropped. Note that the leading
    dimensions of the inputs should be consistent (supports broadcasting).
    """
    def __init__(self, layer_sizes, inputs, K, Dxi=None, **kwargs):
        """
        :param layer_sizes: List of # of units at each layer, including output
           layer.
        :param inputs: List of input tensors. See note of broadcasting.
        :param K: Number of samples to be generated per data point.
        :param kwargs: Other kwargs to be passed to the parent class.
        """
        # TODO: update comments: Added variable latent variable dimension.
        Serializable.quick_init(self, locals())
        self._K = K
        if Dxi is None:
            Dxi = layer_sizes[-1]
        self._Dxi = Dxi

        n_dims = inputs[0].get_shape().ndims

        if K > 1:
            xi_shape = np.ones(n_dims + 1, dtype=np.int32)
            xi_shape[-2:] = (self._K, Dxi)
        else:
            xi_shape = np.ones(n_dims, dtype=np.int32)
            xi_shape[-1:] = Dxi

        expanded_inputs = [tf.expand_dims(t, n_dims-1) for t in inputs]

        # TODO: is it better to use independent random samples for different
        # states?

        xi = tf.random_normal(xi_shape)  # 1 x ... x 1 x K x (output shape)
        expanded_inputs.append(xi)

        super(StochasticNeuralNetwork, self).__init__(
            layer_sizes, expanded_inputs, **kwargs
        )

        self._inputs = inputs  # This will hide the random tensor from inputs.
        self._xi = xi  # For debugging.
