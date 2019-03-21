"""Implements an adapter for DeepMind Control Suite environments."""


# TODO(hartikainen): Need numpy? Import it here.
# import numpy as np

from dm_control import suite

from .softlearning_env import SoftlearningEnv

# TODO(hartikainen): Add information of the available environments.
# See `gym_adapter.py` for example.
DM_CONTROL_ENVIRONMENTS = {}


class DmControlAdapter(SoftlearningEnv):
    """Adapter that implements the SoftlearningEnv for Gym envs."""

    def __init__(self,
                 domain,
                 task,
                 *args,
                 env=None,
                 normalize=True,
                 observation_keys=None,
                 unwrap_time_limit=True,
                 **kwargs):
        assert not args, (
            "Gym environments don't support args. Use kwargs instead.")

        self.normalize = normalize
        self.observation_keys = observation_keys
        self.unwrap_time_limit = unwrap_time_limit

        self._Serializable__initialize(locals())
        super(DmControlAdapter, self).__init__(domain, task, *args, **kwargs)

        if env is None:
            assert (domain is not None and task is not None), (domain, task)
            env = suite.load(
                domain_name=domain,
                task_name=task,
                # TODO(hartikainen): Figure out how to pass kwargs to this guy.
                # Need to split into `task_kwargs`, `environment_kwargs`, and `visualize_reward` bool.
                # Check the suite.load(.) in: 
                # https://github.com/deepmind/dm_control/blob/master/dm_control/suite/__init__.py
            )
        else:
            assert domain is None and task is None, (domain, task)

        assert (self.observation_keys is not None)
        # TODO(hartikainen): Need to handle "Dict" observation space keys here.
        # 1. I believe deepmind control suite only has dict observations, so
        #    might not need to do an if-statement like in GymAdapter. Instead,
        #    always have `self.observation_keys` set.

        # TODO(hartikainen): Do we need to normalize actions? For now, assume
        # we don't.

        self._env = env

    @property
    def observation_space(self):
        # TODO(hartikainen): Figure out how to retrieve the observation space
        # from `self._env`. Something like:
        # observation_space = self._env.observation_space
        # return observation_space
        raise NotImplementedError(
            "TODO(hartikainen): Figure out how to retrieve the observation"
            "space from `self._env`. Something like:")

    @property
    def active_observation_shape(self):
        """Shape for the active observation based on observation_keys."""

        # TODO(hartikainen): Figure out how to parse the dictionary observation
        # space and convert it's shape into a "flattened" shape. Something
        # like:
        # active_size = sum(
        #     np.prod(self._env.observation_space.spaces[key].shape)
        #     for key in observation_keys)
        # active_observation_shape = (active_size, )
        # return active_observation_shape

        raise NotImplementedError(
            "TODO(hartikainen): Figure out how to parse the dictionary "
            "observation space and convert it's shape into a flattened shape.")

    def convert_to_active_observation(self, observation):
        # TODO(hartikainen): Figure out how to convert the dictionary
        # observation into a flattened numpy array. Something like:
        # observation_keys = (
        #     self.observation_keys
        #     or list(self._env.observation_space.spaces.keys()))
        #
        # observation = np.concatenate([
        #     observation[key] for key in observation_keys
        # ], axis=-1)
        #
        # return observation

        raise NotImplementedError(
            "TODO(hartikainen): Figure out how to flatten the observation.")

    @property
    def action_space(self, *args, **kwargs):
        # TODO(hartikainen): Figure out how to get the `action_space` from the
        # `self._env`. In gym this is done by: `self._env.action_space`.
        # Something like:
        # action_space = self._env.action_space
        # if len(action_space.shape) > 1:
        #     raise NotImplementedError(
        #         "Action space ({}) is not flat, make sure to check the"
        #         " implemenation.".format(action_space))
        # return action_space
        raise NotImplementedError(
            "TODO(hartikainen): Figure out how to return env action space.")

    def step(self, action, *args, **kwargs):
        return self._env.step(action, *args, **kwargs)

    def reset(self, *args, **kwargs):
        # TODO(hartikainen): this might of different format than the gym
        # observation. Needs to return a tuple of:
        #
        # (observation, reward, done, info)

        # where `observation` is a dictionary of numpy arrays, `reward` is a
        # scalar, `done` is a boolean, and `info` is a dictionary.
        return self._env.reset(*args, **kwargs)

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self, *args, **kwargs):
        return self._env.close(*args, **kwargs)

    def seed(self, *args, **kwargs):
        return self._env.seed(*args, **kwargs)

    @property
    def unwrapped(self):
        return self._env.unwrapped

    def get_param_values(self, *args, **kwargs):
        raise NotImplementedError

    def set_param_values(self, *args, **kwargs):
        raise NotImplementedError
