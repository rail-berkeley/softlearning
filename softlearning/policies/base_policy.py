from contextlib import contextmanager
from collections import OrderedDict

import numpy as np
from serializable import Serializable


class BasePolicy(Serializable):
    def __init__(self):
        self._deterministic = False

    def reset(self):
        """Reset and clean the policy."""
        raise NotImplementedError

    def actions(self, conditions):
        """Compute (symbolic) actions given conditions (observations)"""
        raise NotImplementedError

    def log_pis(self, conditions, actions):
        """Compute (symbolic) log probs for given observations and actions."""
        raise NotImplementedError

    def actions_np(self, conditions):
        """Compute (numeric) actions given conditions (observations)"""
        raise NotImplementedError

    def log_pis_np(self, conditions, actions):
        """Compute (numeric) log probs for given observations and actions."""
        raise NotImplementedError

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
        diagnostics = OrderedDict({})
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

    def actions_np(self, conditions):
        if self._deterministic:
            return self.deterministic_actions_model.predict(conditions)
        elif self._smoothing_alpha == 0:
            return self.actions_model.predict(conditions)
        else:
            alpha, beta = self._smoothing_alpha, self._smoothing_beta
            raw_latents = self.latents_model.predict(conditions)
            self._smoothing_x = (
                alpha * self._smoothing_x + (1.0 - alpha) * raw_latents)
            latents = beta * self._smoothing_x

            return self.actions_model_for_fixed_latents.predict(
                [*conditions, latents])

    def reset(self):
        self._reset_smoothing_x()
