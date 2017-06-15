from rllab.core.serializable import Serializable
from railrl.core import tf_util
from railrl.core.neuralnet import NeuralNetwork
from rllab.misc.overrides import overrides


class Perceptron(NeuralNetwork):
    """A perceptron, where output = W * input + b"""

    def __init__(
            self,
            name_or_scope,
            input_tensor,
            input_size,
            output_size,
            W_name=tf_util.WEIGHT_DEFAULT_NAME,
            b_name=tf_util.BIAS_DEFAULT_NAME,
            W_initializer=None,
            b_initializer=None,
            **kwargs
    ):
        Serializable.quick_init(self, locals())
        super().__init__(name_or_scope, **kwargs)

        self.input_size = input_size
        self.output_size = output_size
        self.W_name = W_name
        self.b_name = b_name
        self.W_initializer = W_initializer
        self.b_initializer = b_initializer
        super(Perceptron, self).__init__(name_or_scope, **kwargs)
        self._create_network(input_tensor=input_tensor)

    def _create_network_internal(self, input_tensor=None):
        assert input_tensor is not None
        input_tensor = self._process_layer(input_tensor,
                                           scope_name="input_tensor")
        return tf_util.linear(
            input_tensor,
            self.input_size,
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
