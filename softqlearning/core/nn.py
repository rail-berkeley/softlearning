import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.tf.core.parameterized import Parameterized
from softqlearning.misc.mlp import mlp


class ComputationGraph(Parameterized, Serializable):
    """ A wrapper for a tensorflow graph. """

    def __init__(self, scope_name, inputs, output):
        Serializable.quick_init(self, locals())
        super().__init__()

        self._inputs = inputs
        self._output = output
        self._scope_name = scope_name

    @property
    def output(self):
        return self._output

    @property
    def input(self):
        assert len(self._inputs) == 1
        return self._inputs[0]

    @property
    def inputs(self):
        return self._inputs

    @overrides
    def get_params_internal(self, **tags):
        if len(tags) > 0:
            raise NotImplementedError
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 self._scope_name + '/')


class InputBounds(ComputationGraph):
    """
    Modifies the gradient of the given graph ('output') with respect to the
    input so that it always points towards the domain of the input.
    It is assumed that the input domain is equal to the L_\infty unit ball.

    'InputBounds' can be used to implement the SVGD algorithm, which assumes a
    target distribution with infinite action support: 'InputBounds' allows
    actions to temporally violate the boundaries, but the modified gradient will
    eventually bring them back and to satisfy the constraints.
    """
    SLOPE = 10  # This is the new gradient outside the input domain.

    def __init__(self, inp, output):
        """
        :param inp: Input tensor with a constrained domain.
        :param output: Output tensor, whose gradient will be modified.
        """
        scope_name = tf.get_variable_scope().name
        Serializable.quick_init(self, locals())

        violation = tf.maximum(tf.abs(inp) - 1, 0)
        total_violation = tf.reduce_sum(violation, axis=-1, keep_dims=True)

        # Expand the first dimension to match the graph
        # (needed for tf.where which does not support broadcasting).
        expanded_total_violation = total_violation * tf.ones_like(output)

        bounded_output = tf.where(tf.greater(expanded_total_violation, 0),
                                  - self.SLOPE * expanded_total_violation,
                                  output)
        super().__init__(scope_name, [inp], bounded_output)


class NeuralNetwork(ComputationGraph):
    """ Multilayer Perceptron that support broadcasting.

    Supports inputs tensors of rank 2 and rank 3. It is assumed that all but the
    last axis are independent samples, and the last axis is considered as a
    input vector (rank 1 tensor). If multiple inputs are provided, then the
    corresponding vectors are concatenated. Broadcasting is supported for the
    leading dimensions. The leading dimensions of the network output are equal
    to the 'outer product' of the inputs shapes. For example

    input 1 shape: N x K x D1
    input 2 shape: N x 1 x D2

    output shape: N x K x (number of output units))
    """

    def __init__(self, scope_name, layer_sizes, inputs,
                 nonlinearity=tf.nn.relu, output_nonlinearity=None,
                 reuse=False):
        """
        :param scope_name: Variable scope name.
        :param layer_sizes: List of # of units at each layer, including output
           layer.
        :param inputs: List of input tensors. See note of broadcasting.
        :param nonlinearity: Nonlinearity operation for hidden layers.
        :param output_nonlinearity: Nonlinearity operation for output layer.
        :param reuse: If True, will reuse the parameters in the same name scope.
        """
        Serializable.quick_init(self, locals())

        with tf.variable_scope(scope_name, reuse=reuse):

            n_outputs = layer_sizes[-1]
            graph = mlp(inputs,
                        layer_sizes,
                        nonlinearity=nonlinearity,
                        output_nonlinearity=output_nonlinearity)

        super().__init__(scope_name, inputs, graph)


class StochasticNeuralNetwork(NeuralNetwork):
    """
    StochasticNeuralNetwork is like NeuralNetwork, but is inject a random
    input vector with the same dimension as the output to the bottom layer.
    This network can produce multiple random samples at the time, conditioned
    on the other inputs.

    Example:

    input 1 shape: N x D
    output shape: N x K x (number of output units)

    where K is the number of samples.
    """

    def __init__(self, scope_name, layer_sizes, inputs, K, **kwargs):
        """
        :param scope_name: Variable scope name.
        :param layer_sizes: List of # of units at each layer, including output
           layer.
        :param inputs: List of input tensors. See note of broadcasting.
        :param K: Number of samples to be generated in single forward pass.
        :param kwargs: Other kwargs to be passed to the parent class.
        """
        Serializable.quick_init(self, locals())
        self._K = K

        n_dims = inputs[0].get_shape().ndims
        sample_shape = np.ones(n_dims + 1, dtype=np.int32)
        sample_shape[n_dims-1:] = (self._K, layer_sizes[-1])

        expanded_inputs = [tf.expand_dims(t, n_dims-1) for t in inputs]

        xi = tf.random_normal(sample_shape)  # 1 x ... x 1 x K x Do
        expanded_inputs.append(xi)

        super().__init__(scope_name, layer_sizes, expanded_inputs, **kwargs)

        self._inputs = inputs  # This will hide the random tensor from inputs.
        self._xi = xi  # For debugging
