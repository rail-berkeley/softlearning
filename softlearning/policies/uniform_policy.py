from collections import OrderedDict

import tensorflow as tf


class UniformPolicy(object):
    def __init__(self, input_shapes, output_shape, action_range=(-1.0, 1.0)):
        self.inputs = [
            tf.keras.layers.Input(shape=input_shape)
            for input_shape in input_shapes
        ]
        self._action_range = action_range

        x = (tf.keras.layers.Concatenate(axis=-1)(self.inputs)
             if len(self.inputs) > 1
             else self.inputs[0])

        actions = tf.keras.layers.Lambda(
            lambda x: tf.random.uniform(
                (tf.shape(x)[0], output_shape[0]),
                *action_range)
        )(x)

        self.actions_model = tf.keras.Model(self.inputs, actions)

        self.actions_input = tf.keras.Input(shape=output_shape)

        log_pis = tf.keras.layers.Lambda(
            lambda x: tf.tile(tf.log([
                (action_range[1] - action_range[0]) / 2.0
            ])[None], (tf.shape(x)[0], 1))
        )(self.actions_input)

        self.log_pis_model = tf.keras.Model(
            (*self.inputs, self.actions_input), log_pis)

    @property
    def trainable_variables(self):
        return None

    def reset(self):
        pass

    def actions(self, conditions):
        return self.actions_model(conditions)

    def log_pis(self, conditions, actions):
        return self.log_pis_model([*conditions, actions])

    def actions_np(self, conditions):
        return self.actions_model.predict(conditions)

    def log_pis_np(self, conditions, actions):
        return self.log_pis_model.predict([*conditions, actions])

    def get_diagnostics(self, iteration, batch):
        return OrderedDict({})
