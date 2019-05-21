from collections import OrderedDict

import numpy as np
import tensorflow as tf

from .base_policy import BasePolicy
from softlearning.models.utils import create_inputs


class UniformPolicy(BasePolicy):
    def __init__(self,
                 input_shapes,
                 output_shape,
                 *args,
                 action_range=np.array(((-1.0, ), (1.0, ))),
                 preprocessors=None,
                 **kwargs):
        self._Serializable__initialize(locals())

        super(UniformPolicy, self).__init__(*args, **kwargs)

        inputs_flat = create_inputs(input_shapes)

        self.inputs = inputs_flat

        self._action_range = action_range

        x = self.inputs

        batch_size = tf.keras.layers.Lambda(
            lambda x: tf.shape(x)[0]
        )(inputs_flat[0])

        actions = tf.keras.layers.Lambda(
            lambda batch_size: tf.random.uniform(
                (batch_size, output_shape[0]),
                *action_range)
        )(batch_size)

        self.actions_model = tf.keras.Model(self.inputs, actions)

        self.actions_input = tf.keras.Input(shape=output_shape, name='actions')

        log_pis = tf.keras.layers.Lambda(
            lambda x: tf.tile(tf.math.log(
                (action_range[1] - action_range[0]) / 2.0
            )[None], (tf.shape(input=x)[0], 1))
        )(self.actions_input)

        self.log_pis_model = tf.keras.Model(
            (*self.inputs, self.actions_input), log_pis)

    def get_weights(self):
        return []

    def set_weights(self, *args, **kwargs):
        return

    @property
    def trainable_variables(self):
        return []

    def reset(self):
        pass

    def get_diagnostics(self, observations):
        return OrderedDict({})

    def actions(self, observations):
        return self.actions_model(observations)

    def actions_np(self, observations):
        return self.actions_model.predict(observations)
