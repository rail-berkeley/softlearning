import abc
from contextlib import contextmanager
from collections import OrderedDict

import numpy as np
from serializable import Serializable

from softlearning.utils.tensorflow import nest


class BasePolicy(Serializable):
    def __init__(self, observation_keys):
        self._observation_keys = observation_keys
        self._deterministic = False

    @property
    def observation_keys(self):
        return self._observation_keys

    @property
    def input_names(self):
        return self.actions_model.input_names

    def reset(self):
        """Reset and clean the policy."""

    @abc.abstractmethod
    def actions(self, inputs):
        """Compute actions for given inputs (e.g. observations)."""
        raise NotImplementedError

    def action(self, input_):
        """Compute an action for a single input, (e.g. observation)."""
        inputs = nest.map_structure(lambda x: x[None, ...], input_)
        values = self.values(inputs)
        value = nest.map_structure(lambda x: x[0], values)
        return value

    @abc.abstractmethod
    def log_pis(self, inputs, actions):
        """Compute log probabilities for given actions."""
        raise NotImplementedError

    def log_pi(self, *input_):
        """Compute the log probability for a single action."""
        inputs = nest.map_structure(lambda x: x[None, ...], input_)
        values = self.values(*inputs)
        value = nest.map_structure(lambda x: x[0], values)
        return value

    def _filter_observations(self, observations):
        if (isinstance(observations, dict)
            and self._observation_keys is not None):
            observations = type((observations))((
                (key, observations[key])
                for key in self.observation_keys
            ))
        return observations

    @contextmanager
    def set_deterministic(self, deterministic=True):
        """Context manager for changing the determinism of the policy.
        Args:
            set_deterministic (`bool`): Value to set the self._is_deterministic
                to during the context. The value will be reset back to the
                previous value when the context exits.
        """
        was_deterministic = self._deterministic
        self._deterministic = deterministic
        yield
        self._deterministic = was_deterministic

    def get_diagnostics(self, conditions):
        """Return diagnostic information of the policy.

        Arguments:
            conditions: Observations to run the diagnostics for.
        Returns:
            diagnostics: OrderedDict of diagnostic information.
        """
        diagnostics = OrderedDict()
        return diagnostics

    def __getstate__(self):
        state = Serializable.__getstate__(self)
        state['pickled_weights'] = self.get_weights()

        return state

    def __setstate__(self, state):
        Serializable.__setstate__(self, state)
        self.set_weights(state['pickled_weights'])


class LatentSpacePolicy(BasePolicy):
    def __init__(self, *args, smoothing_coefficient=None, **kwargs):
        super(LatentSpacePolicy, self).__init__(*args, **kwargs)

        assert smoothing_coefficient is None or 0 <= smoothing_coefficient <= 1
        self._smoothing_alpha = smoothing_coefficient or 0
        self._smoothing_beta = (
            np.sqrt(1.0 - np.power(self._smoothing_alpha, 2.0))
            / (1.0 - self._smoothing_alpha))
        self._reset_smoothing_x()
        self._smooth_latents = False

    def _reset_smoothing_x(self):
        self._smoothing_x = np.zeros((1, *self._output_shape))

    def actions(self, observations):
        if 0 < self._smoothing_alpha:
            raise NotImplementedError(
                "TODO(hartikainen): Smoothing alpha temporarily dropped on tf2"
                " migration. Should add it back. See:"
                " https://github.com/rail-berkeley/softlearning/blob/46374df0294b9b5f6dbe65b9471ec491a82b6944/softlearning/policies/base_policy.py#L80")

        observations = self._filter_observations(observations)
        if self._deterministic:
            return self.deterministic_actions_model(observations)
        return self.actions_model(observations)

    def log_pis(self, observations, actions):
        observations = self._filter_observations(observations)
        assert not self._deterministic, self._deterministic
        return self.log_pis_model((observations, actions))

    def reset(self):
        self._reset_smoothing_x()
