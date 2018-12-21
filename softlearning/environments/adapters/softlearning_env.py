"""Implements the SoftlearningEnv that is usable in softlearning algorithms."""

from abc import ABCMeta, abstractmethod
from collections import defaultdict

import numpy as np
from serializable import Serializable


class SoftlearningEnv(Serializable, metaclass=ABCMeta):
    """The abstract Softlearning environment class.

    It's an abstract class defining the interface an adapter needs to implement
    in order to function with softlearning algorithms. It closely follows the
    gym.Env, yet that may not be the case in the future.

    The main API methods that users of this class need to know are:

        step
        reset
        render
        close
        seed

    And set the following attributes:

        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards

    The methods are accessed publicly as "step", "reset", etc.. The
    non-underscored versions are wrapper methods to which we may add
    functionality over time.
    """

    # Set this in SOME subclasses
    metadata = {'render.modes': []}
    reward_range = (-float('inf'), float('inf'))
    spec = None

    # Set these in ALL subclasses
    action_space = None
    observation_space = None

    def __init__(self, domain, task, *args, **kwargs):
        """Initialize an environment based on domain and task.
        Keyword Arguments:
        domain   --
        task     --
        *args    --
        **kwargs --
        """
        self._Serializable__initialize(locals())
        self._domain = domain
        self._task = task

    @property
    @abstractmethod
    def observation_space(self):
        raise NotImplementedError

    @property
    def active_observation_shape(self):
        return self.observation_space.shape

    def convert_to_active_observation(self, observation):
        return observation

    @property
    @abstractmethod
    def action_space(self):
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the environment

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
            space.
        """
        raise NotImplementedError

    @abstractmethod
    def render(self, mode='human'):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with
            close (bool): close all open renderings

        Example:

        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}

            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode is 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        raise NotImplementedError

    def render_rollouts(self, paths):
        """Renders past rollouts of the environment."""
        if hasattr(self._env, 'render_rollouts'):
            return self._env.render_rollouts(paths)

        unwrapped_env = self.unwrapped
        if hasattr(unwrapped_env, 'render_rollouts'):
            return unwrapped_env.render_rollouts(paths)

    @abstractmethod
    def close(self):
        """Override _close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        return

    @abstractmethod
    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        pass

    def copy(self):
        """Create a deep copy the environment.

        TODO: Investigate if this can be done somehow else, especially for gym
        envs.
        """
        return Serializable.clone(self)

    @property
    @abstractmethod
    def unwrapped(self):
        """Completely unwrap this env.

        Returns:
            gym.Env: The base non-wrapped gym.Env instance
        """
        return self._env

    def __str__(self):
        return '<{type_name}(domain={domain}, task={task}) <{env}>>'.format(
            type_name=type(self).__name__,
            domain=self._domain,
            task=self._task,
            env=self._env)

    @abstractmethod
    def get_param_values(self):
        raise NotImplementedError

    @abstractmethod
    def set_param_values(self, params):
        raise NotImplementedError

    def get_path_infos(self, paths, *args, **kwargs):
        """Log some general diagnostics from the env infos.

        TODO(hartikainen): These logs don't make much sense right now. Need to
        figure out better format for logging general env infos.
        """
        keys = list(paths[0].get('infos', [{}])[0].keys())

        results = defaultdict(list)

        for path in paths:
            path_results = {
                k: [
                    info[k]
                    for info in path['infos']
                ] for k in keys
            }
            for info_key, info_values in path_results.items():
                info_values = np.array(info_values)
                results[info_key + '-first'].append(info_values[0])
                results[info_key + '-last'].append(info_values[-1])
                results[info_key + '-mean'].append(np.mean(info_values))
                results[info_key + '-median'].append(np.median(info_values))
                if np.array(info_values).dtype != np.dtype('bool'):
                    results[info_key + '-range'].append(np.ptp(info_values))

        aggregated_results = {}
        for key, value in results.items():
            aggregated_results[key + '-mean'] = np.mean(value)

        return aggregated_results
