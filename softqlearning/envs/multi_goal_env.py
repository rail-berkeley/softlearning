from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.spaces.box import Box
from rllab.envs.base import Env

import numpy as np
import matplotlib.pyplot as plt


class MultiGoalEnv(Env, Serializable):
    """
    Move a 2D point mass to one of the goal positions. Cost is the distance to
    the closest goal.

    State: position.
    Action: velocity.
    """
    def __init__(self, goal_reward=10):
        super(MultiGoalEnv, self).__init__()
        Serializable.quick_init(self, locals())

        self.dynamics = PointDynamics(dim=2, sigma=0)
        self.init_mu = np.array((0, 0), dtype=np.float32)
        self.init_sigma = 0
        self.goal_positions = np.array(
            [
                [5, 0],
                [-5, 0],
                [0, 5],
                [0, -5]
            ],
            dtype=np.float32
        )
        self.goal_threshold = 1.
        self.goal_reward = goal_reward
        self.action_cost_coeff = 30.
        self.xlim = (-7, 7)
        self.ylim = (-7, 7)
        self.vel_bound = 1.
        self.reset()
        self.observation = None

        self.fig = None
        self.ax = None
        self.fixed_plots = None
        self.dynamic_plots = []

    @overrides
    def reset(self):
        unclipped_observation = self.init_mu + self.init_sigma * \
            np.random.normal(size=self.dynamics.s_dim)
        o_lb, o_ub = self.observation_space.bounds
        self.observation = np.clip(unclipped_observation, o_lb, o_ub)
        return self.observation

    @overrides
    @property
    def observation_space(self):
        return Box(
            low=np.array((self.xlim[0], self.ylim[0])),
            high=np.array((self.xlim[1], self.ylim[1])),
            shape=None
        )

    @overrides
    @property
    def action_space(self):
        return Box(
            low=-self.vel_bound,
            high=self.vel_bound,
            shape=(self.dynamics.a_dim,)
        )

    def get_current_obs(self):
        return np.copy(self.observation)

    @overrides
    def step(self, action):
        action = action.ravel()

        a_lb, a_ub = self.action_space.bounds
        action = np.clip(action, a_lb, a_ub).ravel()

        next_obs = self.dynamics.forward(self.observation, action)
        o_lb, o_ub = self.observation_space.bounds
        next_obs = np.clip(next_obs, o_lb, o_ub)

        reward = self.compute_reward(self.observation, action)
        cur_position = self.observation
        dist_to_goal = np.amin([
            np.linalg.norm(cur_position - goal_position)
            for goal_position in self.goal_positions
        ])
        done = dist_to_goal < self.goal_threshold
        if done:
            reward += self.goal_reward

        self.observation = np.copy(next_obs)
        return next_obs, reward, done, {'pos': next_obs}

    def init_plot(self, ax):
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.grid(True)
        self.plot_position_cost(ax)

    @staticmethod
    def plot_path(env_info_list, ax, style='b'):
        path = np.concatenate([i['pos'][None] for i in env_info_list], axis=0)
        xx = path[:, 0]
        yy = path[:, 1]
        line, = ax.plot(xx, yy, style)
        return line

    @staticmethod
    def plot_paths(paths, ax):
        line_lst = []
        for path in paths:
            positions = path["env_infos"]["pos"]
            xx = positions[:, 0]
            yy = positions[:, 1]
            line_lst += ax.plot(xx, yy, 'b')
        return line_lst

    def compute_reward(self, observation, action):
        # penalize the L2 norm of acceleration
        # noinspection PyTypeChecker
        action_cost = np.sum(action ** 2) * self.action_cost_coeff

        # penalize squared dist to goal
        cur_position = observation
        # noinspection PyTypeChecker
        goal_cost = np.amin([
            np.sum((cur_position - goal_position) ** 2)
            for goal_position in self.goal_positions
        ])

        # penalize staying with the log barriers
        costs = [action_cost, goal_cost]
        reward = -np.sum(costs)
        return reward

    def plot_position_cost(self, ax):
        delta = 0.01
        x_min, x_max = tuple(1.1 * np.array(self.xlim))
        y_min, y_max = tuple(1.1 * np.array(self.ylim))
        X, Y = np.meshgrid(
            np.arange(x_min, x_max, delta),
            np.arange(y_min, y_max, delta)
        )
        goal_costs = np.amin([
            (X - goal_x) ** 2 + (Y - goal_y) ** 2
            for goal_x, goal_y in self.goal_positions
        ], axis=0)
        costs = goal_costs

        contours = ax.contour(X, Y, costs, 20)
        ax.clabel(contours, inline=1, fontsize=10, fmt='%.0f')
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        goal = ax.plot(self.goal_positions[:, 0],
                       self.goal_positions[:, 1], 'ro')
        return [contours, goal]

    def get_param_values(self):
        return None

    def set_param_values(self, params):
        pass

    @overrides
    def log_diagnostics(self, paths):
        n_goal = len(self.goal_positions)
        goal_reached = [False] * n_goal

        for path in paths:
            last_obs = path["observations"][-1]
            for i, goal in enumerate(self.goal_positions):
                if np.linalg.norm(last_obs - goal) < self.goal_threshold:
                    goal_reached[i] = True

        logger.record_tabular('env:goals_reached', goal_reached.count(True))

    def horizon(self):
        return None

    @overrides
    def render(self, close=False):
        if self.fig is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            plt.axis('equal')

        if self.fixed_plots is None:
            self.fixed_plots = self.plot_position_cost(self.ax)

        [o.remove() for o in self.dynamic_plots]

        x, y = self.observation
        point = self.ax.plot(x, y, 'b*')
        self.dynamic_plots = point

        if close:
            self.fixed_plots = None

        plt.pause(0.001)
        plt.draw()


class PointDynamics(object):
    """
    State: position.
    Action: velocity.
    """
    def __init__(self, dim, sigma):
        self.dim = dim
        self.sigma = sigma
        self.s_dim = dim
        self.a_dim = dim

    def forward(self, state, action):
        mu_next = state + action
        state_next = mu_next + self.sigma * \
            np.random.normal(size=self.s_dim)
        return state_next
