"""Implements a GymAdapter that converts Gym envs in a SoftlearningEnv."""

import gym

from rllab.core.serializable import Serializable

from .softlearning_env import SoftlearningEnv


GYM_ENVIRONMENTS = {
    'swimmer': {
        'default': 'Swimmer-v2'
    },
    'ant': {
        'default': 'Ant-v2'
    },
    'humanoid': {
        'default': 'Humanoid-v2',
        'standup': 'HumanoidStandup-v2'
    },
    'hopper': {
        'default': 'Hopper-v2'
    },
    'half-cheetah': {
        'default': 'HalfCheetah-v2'
    },
    'walker': {
        'default': 'Walker2d-v2'
    },
}


class GymAdapter(SoftlearningEnv):
    """Adapter to convert Gym environment into standard."""

    def __init__(self, domain, task, *args, normalize=True, **kwargs):
        Serializable.quick_init(self, locals())
        super(GymAdapter, self).__init__(domain, task, *args, **kwargs)

        env_name = GYM_ENVIRONMENTS[domain][task]
        env = gym.envs.make(env_name)
        # Remove the TimeLimit wrapper that sets 'done = True' when
        # the time limit specified for each environment has been passed and
        # therefore the environment is not Markovian (terminal condition
        # depends on time rather than state).
        self._env = env.env

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self, *args, **kwargs):
        return self._env.action_space

    def step(self, action, *args, **kwargs):
        # TODO(hartikainen): refactor this to always return OrderedDict,
        # such that the observation for all the envs is consistent. Right now
        # Some of the gym envs return np.array whereas other return dict.
        #
        # Something like:
        # observation = OrderedDict()
        # observation['observation'] = env.step(action, *args, **kwargs)
        # return observation

        return self._env.step(action, *args, **kwargs)

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self, *args, **kwargs):
        return self._env.close(*args, **kwargs)

    def seed(self, *args, **kwargs):
        return self._env.seed(*args, **kwargs)

    def unwrapped(self, *args, **kwargs):
        return self._env.unwrapped(*args, **kwargs)

    def copy(self, *args, **kwargs):
        return self._env.copy(*args, **kwargs)

    def get_param_values(self, *args, **kwargs):
        raise NotImplementedError

    def set_param_values(self, *args, **kwargs):
        raise NotImplementedError
