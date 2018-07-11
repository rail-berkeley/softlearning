from rllab.core.serializable import Serializable

from rllab.misc.overrides import overrides
from sandbox.rocky.tf.policies.base import Policy

import numpy as np


class UniformPolicy(Policy, Serializable):
    """Fixed policy that samples actions uniformly randomly.

    Used for an initial exploration period instead of an undertrained policy.
    """
    def __init__(self, env_spec):
        Serializable.quick_init(self, locals())
        self._Da = env_spec.action_space.flat_dim

        super(UniformPolicy, self).__init__(env_spec)

    @overrides
    def get_action(self, observation, with_log_pis=False, with_raw_actions=False):
        """Get single actions for the observation.

        Assumes action spaces are normalized to be the interval [-1, 1]."""
        action = np.random.uniform(-1., 1., self._Da)
        return_list = [action]
        if with_log_pis:
            return_list.append(0.)
        if with_raw_actions:
            # atanh is unstable when actions are too close to +/- 1, but seems stable
            # at least between -1 + 1e-10 and 1 - 1e-10, so we shouldn't need to worry.
            return_list.append(np.arctanh(action))
        return return_list + [{}]

    @overrides
    def get_actions(self, observations):
        pass

    @overrides
    def log_diagnostics(self, paths):
        pass

    @overrides
    def get_params_internal(self, **tags):
        pass
