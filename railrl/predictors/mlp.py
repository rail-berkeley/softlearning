"""
A series of perceptrons. Implemented mostly for unittests.
"""
from rllab.core.serializable import Serializable
from railrl.core import tf_util
from railrl.core.neuralnet import NeuralNetwork
from railrl.predictors.perceptron import Perceptron
from rllab.misc.overrides import overrides


class Mlp(NeuralNetwork):
    """A multi-layer perceptron"""

    def __init__(
            self,
            name_or_scope,
            input_tensor,
            input_size,
            output_size,
            hidden_sizes,
            W_name=tf_util.WEIGHT_DEFAULT_NAME,
            b_name=tf_util.BIAS_DEFAULT_NAME,
            W_initializer=None,
            b_initializer=None,
            **kwargs
    ):
        Serializable.quick_init(self, locals())
        super().__init__(name_or_scope, **kwargs)
        assert len(hidden_sizes) > 0

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.W_name = W_name
        self.b_name = b_name
        self.W_initializer = W_initializer
        self.b_initializer = b_initializer
        super(Mlp, self).__init__(name_or_scope, **kwargs)
        self._create_network(input_tensor=input_tensor)

    def _create_network_internal(self, input_tensor=None):
        assert input_tensor is not None
        input_tensor = self._process_layer(input_tensor,
                                           scope_name="input_tensor")
        in_size = self.input_size
        for layer, next_size in enumerate(self.hidden_sizes):
            p = Perceptron(
                'p{0}'.format(layer),
                input_tensor,
                in_size,
                next_size,
                W_name=self.W_name,
                b_name=self.b_name,
                W_initializer=self.W_initializer,
                b_initializer=self.b_initializer,
                batch_norm_config=self._batch_norm_config,
            )
            input_tensor = self._add_subnetwork_and_get_output(p)
            input_tensor = self._process_layer(input_tensor)
            in_size = next_size
        return tf_util.linear(
            input_tensor,
            in_size,
            self.output_size,
            W_name=self.W_name,
            b_name=self.b_name,
            W_initializer=self.W_initializer,
            b_initializer=self.b_initializer,
        )

    @property
    @overrides
    def _input_name_to_values(self):
        return dict(
            input_tensor=None,
        )
