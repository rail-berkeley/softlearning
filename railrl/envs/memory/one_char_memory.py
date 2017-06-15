import numpy as np
from random import randint
from rllab.envs.base import Env
from rllab.misc import special
from rllab.spaces.discrete import Discrete
from railrl.envs.supervised_learning_env import SupervisedLearningEnv


class OneCharMemory(Env, SupervisedLearningEnv):
    """
    A simple env whose output is a value `X` the first time step, followed by a
    fixed number of zeros.

    The reward is 1 for all steps if the agent outputs zero except at the end,
    where the agent gets a large reward if `X` is returned correctly.

    Both the actions and observations are represented as one-hot vectors.
    There are `n` different values that `X` can take on (excluding 0),
    so the one-hot vector's dimension is n+1.
    """

    def __init__(self, n=4, num_steps=100, reward_for_remembering=1000):
        """
        :param n: Number of different values that could be returned
        :param num_steps: How many steps the agent needs to remember.
        :param reward_for_remembering: The reward for remembering the number.
        """
        super().__init__()
        self.num_steps = num_steps
        self.n = n
        self._action_space = Discrete(self.n + 1)
        self._observation_space = Discrete(self.n + 1)
        self._t = 1
        self._reward_for_remembering = reward_for_remembering

        self._target = None
        self._next_obs = None

    def step(self, action):
        # flatten = to one hot...not sure why it was given that name.
        observation = self._get_next_observation()
        self._next_obs = 0

        done = self._t == self.num_steps
        self._t += 1

        if done:
            reward = self._reward_for_remembering * int(
                self._observation_space.unflatten(action) == self._target
            )
        else:
            reward = int(
                self._observation_space.unflatten(action) == 0
            )
        info = {'target': self.n}
        return observation, reward, done, info

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self.num_steps

    def reset(self):
        self._target = randint(1, self.n)
        self._next_obs = self._target
        self._t = 1
        return self._get_next_observation()

    def _get_next_observation(self):
        return self._observation_space.flatten(self._next_obs)

    @property
    def observation_space(self):
        return self._observation_space

    def get_batch(self, batch_size):
        targets = np.random.randint(
            low=1,
            high=self.n,
            size=batch_size,
        )
        onehot_targets = special.to_onehot_n(targets, self.feature_dim)
        X = np.zeros((batch_size, self.sequence_length, self.feature_dim))
        X[:, :, 0] = 1  # make the target 0
        X[:, 0, :] = onehot_targets
        Y = np.zeros((batch_size, self.sequence_length, self.target_dim))
        Y[:, :, 0] = 1  # make the target 0
        Y[:, -1, :] = onehot_targets
        return X, Y

    @property
    def feature_dim(self):
        return self.n + 1

    @property
    def target_dim(self):
        return self.n + 1

    @property
    def sequence_length(self):
        return self.horizon
