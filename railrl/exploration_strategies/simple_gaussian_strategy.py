from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.exploration_strategies.base import ExplorationStrategy
import numpy as np


class SimpleGaussianStrategy(ExplorationStrategy, Serializable):
    """
    This strategy adds a constant Gaussian noise to the action taken by the
    deterministic policy.

    This is different from rllab's GaussianStrategy class in that the sigma
    does not decay over time.
    """

    def __init__(self, env_spec, sigma=1.0):
        assert isinstance(env_spec.action_space, Box)
        assert len(env_spec.action_space.shape) == 1
        Serializable.quick_init(self, locals())
        super().__init__()
        self._sigma = sigma
        self._action_space = env_spec.action_space

    def get_action(self, t, observation, policy, **kwargs):
        action, agent_info = policy.get_action(observation)
        return np.clip(action + np.random.normal(size=len(action))*self._sigma,
                       self._action_space.low,
                       self._action_space.high)
