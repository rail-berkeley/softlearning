import abc
import tensorflow as tf
from typing import Iterable

from railrl.core import tf_util
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.core.parameterized import Parameterized

ALLOWABLE_TAGS = ['regularizable']
_TRAINING_OUTPUT_MODE = "training_output_mode"
_EVAL_OUTPUT_MODE = "eval_output_mode"


class NeuralNetwork(Parameterized, Serializable):
    """
    Any neural network.
    """

    def __init__(
            self,
            name_or_scope,
            batch_norm_config=None,
            reuse=False,
            **kwargs
    ):
        """

        :param name_or_scope: Name (string) for creating this network. This
        should only be name of this current network, and not the full scope
        name, i.e. `bar` not `foo/bar`.
        :param batch_norm_config: Config for batch_norm. If set, batch_norm
        is enabled.
        :param reuse: Reuse variables when creating this network.
        :param kwargs:
        """
        super().__init__(**kwargs)
        Serializable.quick_init(self, locals())
        assert isinstance(name_or_scope, str)
        assert '/' not in name_or_scope
        self.scope_name = name_or_scope
        self._batch_norm = batch_norm_config is not None
        self._batch_norm_config = batch_norm_config
        self._reuse = reuse
        self._bn_stat_update_ops = []
        self._sess = None
        self._output = None
        self._output = None
        self._eval_output = None
        self._training_output = None
        self._full_scope_name = None
        self._subnetwork_list = []
        self._is_bn_in_training_mode = False

    @property
    def full_scope_name(self):
        """
        The fully qualified scope name of this network when it was created, e.g.
        `foo/bar/baz`, whereas `self.scope` would return `baz`.
        :return:
        """
        return self._full_scope_name

    def _create_network(self, **inputs):
        """
        This method should be called by subclasses after the super call is made.

        :param inputs: named Tensors
        :return: None
        """
        if self._batch_norm:
            # TODO(vpong): This flag needs to somehow propagate to sub-networks
            self._switch_to_bn_training_mode_on()
            with tf.variable_scope(self.scope_name, reuse=self._reuse):
                self._training_output = self._create_network_internal(**inputs)
            self._switch_to_bn_training_mode_off()
            with tf.variable_scope(self.scope_name, reuse=True) as scope:
                self._eval_output = self._create_network_internal(**inputs)
        else:
            with tf.variable_scope(self.scope_name, reuse=self._reuse) as scope:
                self._eval_output = self._create_network_internal(**inputs)
                self._training_output = self._eval_output
        # It's important to make this equality and not += since a network
        # may be composed of many sub-networks. Doing += would cause ops
        # to be added twice (once in the sub-network's constructor,
        # and again in the parent constructor).
        self._bn_stat_update_ops = (
            tf_util.get_batch_norm_update_pop_stats_ops(scope=scope)
        )
        self._full_scope_name = scope.name
        self.switch_to_eval_mode()

    def _switch_to_bn_training_mode_on(self):
        self._is_bn_in_training_mode = True

    def _switch_to_bn_training_mode_off(self):
        self._is_bn_in_training_mode = False

    def _iter_sub_networks(self):
        """
        Iterate through sub-networks.
        :return:
        """
        for child in self._subnetworks:
            yield child
            for sub_child in child._iter_sub_networks():
                yield sub_child

    def switch_to_training_mode(self):
        """
        Output will be the output for training.

        Note that only the output of this network will switch. If this
        network is composed of sub-networks, then those network's outputs
        will be independently set.
        :return: None
        """
        self._output = self._training_output

    def switch_to_eval_mode(self):
        """
        Output will be the output for eval. By default, a network is in eval
        mode.

        Note that only the output of this network will switch. If this
        network is composed of sub-networks, then those network's outputs
        will be independently set.
        :return: None
        """
        self._output = self._eval_output

    @property
    def sess(self):
        if self._sess is None:
            self._sess = tf.get_default_session()
        return self._sess

    @sess.setter
    def sess(self, value):
        self._sess = value

    @property
    def output(self):
        """
        :return: Tensor/placeholder/op. Output of this network.
        """
        return self._output

    @property
    def training_output(self):
        """
        :return: Tensor/placeholder/op. Training output of this network.
        """
        return self._training_output

    @property
    def eval_output(self):
        """
        :return: Tensor/placeholder/op. Eval output of this network.
        """
        return self._eval_output

    def _process_layer(self, previous_layer, scope_name="process_layer"):
        """
        This should be done called between every layer, i.e.

        a = self.process_layer(linear(x))
        b = self.process_layer(linear(relu(a)))
        output = linear(relu(b))

        This method should NOT be called on the output.

        If batch norm is disabled, this just returns `previous_layer`
        immediately.

        If batch norm is enabled, this returns a layer with bn enabled,
        either for training or eval based on self._is_bn_in_training_mode

        :param previous_layer: Either the input layer or the output to a
        previous call to `_process_layer`
        :param scope_name: Scope name for any variables that may be created
        while processing the layer
        :return: If batch norm is disabled, this returns previous_layer.
        Otherwise, it returns a tuple (batch norm'd layer for training,
        batch norm'd layer for eval)
        """
        if not self._batch_norm:
            return previous_layer

        with tf.variable_scope(scope_name):
            processed_layer, _ = tf_util.batch_norm(
                previous_layer,
                is_training=self._is_bn_in_training_mode,
                batch_norm_config=self._batch_norm_config,
            )
            return processed_layer

    def _add_subnetwork_and_get_output(self, subnetwork):
        """
        Any time a network has a sub-network, it needs to call this method
        to gets that subnetwork's output. e.g.

        ```
        def _create_internal_network(...):
            n1 = Network1()
            out_of_n1 = self._process_network_and_get_output(n1)

            # out_of_n1 is a Tensor that can be used
            # you probably want to call
            out_of_n1 = self._process_layer(out_of_n1)
        ```
        """
        if (subnetwork.full_scope_name not in
                [n.full_scope_name for n in self._subnetwork_list]):
            self._subnetwork_list.append(subnetwork)

        if not self._batch_norm:
            return subnetwork.output

        if self._is_bn_in_training_mode:
            return subnetwork.training_output
        else:
            return subnetwork.eval_output

    @overrides
    def get_params_internal(self, **tags):
        for key in tags.keys():
            if key not in ALLOWABLE_TAGS:
                raise KeyError(
                    "Tag not allowed: {0}. Allowable tags: {1}".format(
                        key,
                        ALLOWABLE_TAGS))
        # TODO(vpong): This code is pretty messy. Clean it up
        filters = []
        if 'regularizable' in tags:
            regularizable_vars = tf_util.get_regularizable_variables(
                self.full_scope_name)
            if tags['regularizable']:
                reg_filter = lambda v: v in regularizable_vars
            else:
                reg_filter = lambda v: v not in regularizable_vars
            filters.append(reg_filter)

        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      self.full_scope_name)
        if self._batch_norm:
            variables += tf_util.get_untrainable_batch_norm_vars(
                self.full_scope_name
            )
        return list(filter(lambda v: all(f(v) for f in filters), variables))

    def get_copy(self, **kwargs):
        return Serializable.clone(
            self,
            **kwargs
        )

    def get_weight_tied_copy(self, **inputs):
        """
        Return a weight-tied copy of the network. Replace the action or
        observation to the network for the returned network.

        :param inputs: Dictionary, of the form
        {
            'input_x': self.input_x,
            'input_y': self.input_y,
        }

        :return: StateNetwork copy with weights tied to this StateNetwork.
        """
        assert len(inputs) > 0
        for input_name, input_value in self._input_name_to_values.items():
            if input_name not in inputs:
                inputs[input_name] = input_value
        return self.get_copy(
            reuse=True,
            **inputs
        )

    def setup_serialization(self, init_locals):
        Serializable.quick_init(self, init_locals)

    @property
    def batch_norm_update_stats_op(self):
        return self._bn_stat_update_ops

    @property
    def _subnetworks(self) -> Iterable['NeuralNetwork']:
        """
        If this network is built with sub-networks (i.e.
        _create_network_internal creates some NeuralNetwork), then this must
        iterate through them uniquely.
        :return:
        """
        return self._subnetwork_list

    @abc.abstractmethod
    def _create_network_internal(self, **inputs):
        """
        This function constructs the network.

        If you plan on using batch norm:
            - Between each layer, call `layer = self._process_layer(layer)`.
            - Between each subnetwork, call
            `output = self._add_subnetwork_and_get_output(subnetwork)`.
            - See the documentation of those functions for more detail.

        :param inputs: Tensor inputs to the network by name.
        :return: Tensor, output of network
        """
        pass

    @property
    @abc.abstractmethod
    def _input_name_to_values(self):
        """
        Return a dictionary describing what inputs are and their current values.

        :return: Return a dictionary of the form
        {
            'input_x': self.input_x,
            'input_y': self.input_y,
        }

        This will be the input to get_weight_tied_copy for inputs not
        specified.
        """
        pass

    ############################################
    # Helper methods mostly just for debugging #
    ############################################
    def _get_all_variable_names(self):
        return [p.name for p in self.get_params_internal()]
