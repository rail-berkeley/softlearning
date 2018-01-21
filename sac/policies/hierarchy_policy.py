"""Hierarchy policy"""

from contextlib import contextmanager
import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable

from sac.distributions import RealNVPBijector
from sac.policies import NNPolicy, RealNVPPolicy

class HierarchyPolicy(RealNVPPolicy):
    def __init__(self,
                 env_spec,
                 low_level_policy=None,
                 control_interval=10,
                 mode="train",
                 squash=True,
                 real_nvp_config=None,
                 observations_preprocessor=None,
                 name="hierarchy_policy"):
        """Initialize Hierarchy policy.

        Args:
            env_spec (`rllab.EnvSpec`): Specification of the environment
                to create the policy for.
        """
        assert low_level_policy is not None
        Serializable.quick_init(self, locals())

        self._env_spec = env_spec
        self._low_level_policy = low_level_policy
        self._control_interval = control_interval
        self._squash = squash
        self._mode = mode
        self._real_nvp_config = real_nvp_config
        self._observations_preprocessor = observations_preprocessor

        self._Da = env_spec.action_space.flat_dim
        # _Ds for high-level state dimension
        self._Ds = env_spec.observation_space.flat_dim
        # _Ds_L for low-level state dimension
        self._Ds_L = low_level_policy._Ds
        self._fixed_h = None
        self._is_deterministic = False

        self.name = name
        self.build()

        NNPolicy.__init__(
            self,
            env_spec,
            self._observations_ph,
            tf.tanh(self._actions) if squash else self._actions,
            scope_name=name)


    def actions_for(self, observations, name=None, reuse=tf.AUTO_REUSE,
                    stop_gradient=True):
        # TODO: Our actions should be modulated by `self._control_interval`
        high_level_actions = super().actions_for(observations,
                                                 name=name,
                                                 reuse=reuse,
                                                 stop_gradient=stop_gradient)

        low_level_observations = observations[:, :self._Ds_L]
        actions = self._low_level_policy.bijector.forward(
            high_level_actions, conditions=low_level_observations)

        return actions
