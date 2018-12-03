from contextlib import contextmanager
from collections import OrderedDict

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
