"""Hierarchy policy"""

from contextlib import contextmanager
import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable

from sac.distributions import RealNVPBijector
from sac.policies import NNPolicy

class HierarchyPolicy(NNPolicy, Serializable):
    def __init__(self,
                 env_spec,
                 low_level_policy=None,
                 control_interval=10,
                 mode="train",
                 squash=True,
                 real_nvp_config=None,
                 observations_preprocessor=None,
                 name="policy"):
        """Initialize Hierarchy policy.

        Args:
            env_spec (`rllab.EnvSpec`): Specification of the environment
                to create the policy for.
        """
        # assert low_level_policy is not None
        Serializable.quick_init(self, locals())

        self._env_spec = env_spec
        self._low_level_policy = low_level_policy
        self._squash = squash
        self._mode = mode
        self._real_nvp_config = real_nvp_config
        self._observations_preprocessor = observations_preprocessor

        self._Da = env_spec.action_space.flat_dim
        self._Ds = env_spec.observation_space.flat_dim
        self._fixed_h = None
        self._is_deterministic = False

        self.name = name
        self.build()

        super().__init__(
            env_spec,
            self._observations_ph,
            tf.tanh(self._actions) if squash else self._actions,
            scope_name='policy'
        )

    def actions_for(self, observations):
        """TODO: implement"""
        pass

    def log_pi_for(self, observations):
        """TODO: implement"""
        pass

    def build(self):
        """TODO: implement"""
        pass

    def get_action(self, observation):
        """Sample single action based on the observations."""
        return self.get_actions(observation[None])[0], {}

    def get_actions(self, observations):
        """Sample batch of actions based on the observations."""
        pass

    @contextmanager
    def deterministic(self, set_deterministic=True):
        """Context manager for changing the determinism of the policy.

        See `self.get_action` for further information about the effect of
        self._is_deterministic.

        Args:
            set_deterministic (`bool`): Value to set the self._is_deterministic
            to during the context. The value will be reset back to the previous
            value when the context exits.
        """
        was_deterministic = self._is_deterministic
        self._is_deterministic = set_deterministic
        yield
        self._is_deterministic = was_deterministic

    @contextmanager
    def fix_h(self, h=None):
        """TODO: implement"""
        pass

    def get_params_internal(self, **tags):
        if tags: raise NotImplementedError
        return tf.trainable_variables(scope=self.name)

    def reset(self, dones=None):
        """TODO: implement"""
        pass

    def log_diagnostics(self, batch):
        """Record diagnostic information to the logger.

        TODO: implement
        """
        pass
