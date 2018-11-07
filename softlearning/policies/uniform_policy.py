from collections import OrderedDict

import tensorflow as tf

from serializable import Serializable

from sandbox.rocky.tf.policies.base import Policy

import numpy as np


class UniformPolicy(Policy, Serializable):
    """Fixed policy that samples actions uniformly randomly.

    Used for an initial exploration period instead of an undertrained policy.
    """
    def __init__(self, observation_shape, action_shape):
        self._Serializable__initialize(locals())

        assert len(observation_shape) == 1, observation_shape
        self._Ds = observation_shape[0]
        assert len(action_shape) == 1, action_shape
        self._Da = action_shape[0]

        super(UniformPolicy, self).__init__(env_spec=None)

    def get_action(self,
                   observation,
                   with_log_pis=False,
                   with_raw_actions=False):
        """Get single actions for the observation.

        Assumes action spaces are normalized to be the interval [-1, 1]."""
        action = np.random.uniform(-1., 1., self._Da)
        outputs = (
            action,
            0.0 if with_log_pis else None,
            # atanh is unstable when actions are too close to +/- 1, but seems
            # stable at least between -1 + 1e-10 and 1 - 1e-10, so we shouldn't
            # need to worry.
            np.arctanh(action) if with_raw_actions else None)

        return outputs, {}

    def get_actions(self, observations, *args, **kwargs):
        actions, log_pis, raw_actions = None, None, None
        agent_info = {}
        return (actions, log_pis, raw_actions), agent_info

    def log_diagnostics(self, paths):
        pass

    def get_params_internal(self, **tags):
        pass


class UniformPolicyV2(object):
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
