from rllab.core.serializable import Serializable

from rllab.misc.overrides import overrides
from sandbox.rocky.tf.policies.base import Policy

import numpy as np


class UniformPolicy(Policy, Serializable):
    def __init__(self, env_spec):
        Serializable.quick_init(self, locals())
        self._Da = env_spec.action_space.flat_dim

        super(UniformPolicy, self).__init__(env_spec)

    @overrides
    def get_action(self, observation):
        return np.random.uniform(-1., 1., self._Da), None # random from -1 to 1, which is same as what is squashed

    @overrides
    def get_actions(self, observations):
        return # don't need this
        feeds = {self._obs_pl: observations}
        actions = tf.get_default_session().run(self._action, feeds)
        return actions

    @overrides
    def log_diagnostics(self, paths):
        pass

    @overrides
    def get_params_internal(self, **tags):
        pass
