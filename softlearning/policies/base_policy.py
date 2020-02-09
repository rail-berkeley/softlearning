import abc
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tree

from softlearning.models.utils import create_inputs
from softlearning.utils.tensorflow import cast_and_concat, apply_preprocessors


class BasePolicy:
    def __init__(self,
                 input_shapes,
                 output_shape,
                 observation_keys,
                 preprocessors=None,
                 name='policy'):
        self._input_shapes = input_shapes
        self._output_shape = output_shape
        self._observation_keys = observation_keys
        self._inputs = create_inputs(input_shapes)
        self._preprocessors = preprocessors
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def preprocessors(self):
        return self._preprocessors

    @property
    def inputs(self):
        return self._inputs

    @property
    def observation_keys(self):
        return self._observation_keys

    @property
    def input_names(self):
        return self.actions_model.input_names

    def reset(self):
        """Reset and clean the policy."""

    def get_weights(self):
        return []

    def set_weights(self, *args, **kwargs):
        return []

    @property
    def trainable_variables(self):
        return self.trainable_weights

    @property
    def non_trainable_variables(self):
        return self.non_trainable_weights

    @property
    def trainable_weights(self):
        return []

    @property
    def non_trainable_weights(self):
        return []

    @abc.abstractmethod
    def actions(self, inputs):
        """Compute actions for given inputs (e.g. observations)."""
        raise NotImplementedError

    def action(self, input_):
        """Compute an action for a single input, (e.g. observation)."""
        inputs = tree.map_structure(lambda x: x[None, ...], input_)
        actions = self.actions(inputs)
        action = tree.map_structure(lambda x: x[0], actions)
        return action

    @abc.abstractmethod
    def log_probs(self, inputs, actions):
        """Compute log probabilities for given actions."""
        raise NotImplementedError

    def log_prob(self, *input_):
        """Compute the log probability for a single action."""
        inputs = tree.map_structure(lambda x: x[None, ...], input_)
        log_probs = self.values(*inputs)
        log_prob = tree.map_structure(lambda x: x[0], log_probs)
        return log_prob

    @abc.abstractmethod
    def probs(self, inputs, actions):
        """Compute probabilities for given actions."""
        raise NotImplementedError

    @abc.abstractmethod
    def prob(self, *input_):
        """Compute the probability for a single action."""
        inputs = tree.map_structure(lambda x: x[None, ...], input_)
        probs = self.values(*inputs)
        prob = tree.map_structure(lambda x: x[0], probs)
        return prob

    def _preprocess_inputs(self, inputs):
        if self.preprocessors is None:
            preprocessors = tree.map_structure(lambda x: None, inputs)
        else:
            preprocessors = self.preprocessors

        preprocessed_inputs = apply_preprocessors(preprocessors, inputs)

        preprocessed_inputs = tf.keras.layers.Lambda(
            cast_and_concat
        )(preprocessed_inputs)

        return preprocessed_inputs

    def _filter_observations(self, observations):
        if (isinstance(observations, dict)
            and self._observation_keys is not None):
            observations = type(observations)((
                (key, observations[key])
                for key in self.observation_keys
            ))
        return observations

    def get_diagnostics(self, inputs):
        """Return diagnostic information of the policy.

        Arguments:
            conditions: Observations to run the diagnostics for.
        Returns:
            diagnostics: OrderedDict of diagnostic information.
        """
        diagnostics = OrderedDict()
        return diagnostics

    def get_diagnostics_np(self, *args, **kwargs):
        diagnostics = self.get_diagnostics(*args, **kwargs)
        diagnostics_np = tree.map_structure(lambda x: x.numpy(), diagnostics)
        return diagnostics_np

    def get_config(self):
        config = {
            'input_shapes': self._input_shapes,
            'output_shape': self._output_shape,
            'observation_keys': self._observation_keys,
            # 'preprocessors': preprocessors.serialize(self._preprocessors),
            'preprocessors': self._preprocessors,
            'name': self._name,
        }
        return config


class ContinuousPolicy(BasePolicy):
    def __init__(self,
                 action_range,
                 *args,
                 squash=True,
                 **kwargs):
        assert (np.all(action_range == np.array([[-1], [1]]))), (
            "The action space should be scaled to (-1, 1)."
            " TODO(hartikainen): We should support non-scaled actions spaces.")
        self._action_range = action_range
        self._squash = squash
        self._action_post_processor = {
            True: tfp.bijectors.Tanh(),
            False: tfp.bijectors.Identity(),
        }[squash]

        return super(ContinuousPolicy, self).__init__(*args, **kwargs)

    def get_config(self):
        base_config = super(ContinuousPolicy, self).get_config()
        config = {
            **base_config,
            'action_range': self._action_range,
            'squash': self._squash,
        }
        return config


class LatentSpacePolicy(ContinuousPolicy):
    def __init__(self, *args, smoothing_coefficient=None, **kwargs):
        super(LatentSpacePolicy, self).__init__(*args, **kwargs)

        assert smoothing_coefficient is None or 0 <= smoothing_coefficient <= 1

        if smoothing_coefficient is not None and 0 < smoothing_coefficient:
            raise NotImplementedError(
                "TODO(hartikainen): Latent smoothing temporarily dropped on tf2"
                " migration. Should add it back. See:"
                " https://github.com/rail-berkeley/softlearning/blob/46374df0294b9b5f6dbe65b9471ec491a82b6944/softlearning/policies/base_policy.py#L80")

        self._smoothing_coefficient = smoothing_coefficient
        self._smoothing_alpha = smoothing_coefficient or 0
        self._smoothing_beta = (
            np.sqrt(1.0 - np.power(self._smoothing_alpha, 2.0))
            / (1.0 - self._smoothing_alpha))
        self._reset_smoothing_x()
        self._smooth_latents = False

    def _reset_smoothing_x(self):
        self._smoothing_x = np.zeros((1, *self._output_shape))

    @tf.function(experimental_relax_shapes=True)
    def actions(self, observations):
        observations = self._filter_observations(observations)

        shifts, scales = self.shift_and_scale_model(observations)
        batch_shape = tf.shape(tree.flatten(observations)[0])[:-1]
        actions = self.action_distribution.sample(
            batch_shape,
            bijector_kwargs={'scale': {'scale': scales},
                             'shift': {'shift': shifts}})

        return actions

    def reset(self):
        self._reset_smoothing_x()

    def get_config(self):
        base_config = super(LatentSpacePolicy, self).get_config()
        config = {
            **base_config,
            'smoothing_coefficient': self._smoothing_coefficient,
        }
        return config
