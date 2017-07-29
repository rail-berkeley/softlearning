from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from rllab.envs.base import Env
from rllab.spaces.box import Box

import numpy as np


class MinimazeEnv(Env, Serializable):
    """
    Represents a 2D point mass (double integrator) in a maze. Task is to find a
    goal location. The reward function is proportional to the quadratic distance
    to the goal. There can be several goals, and only the closest goal
    contributes the total reward.

    state: position and velocity in 2D
    action: force
    """
    def __init__(self, maze_id=0, goals=None, dynamics=None):
        super(MinimazeEnv, self).__init__()
        Serializable.quick_init(self, locals())

        self._max_force = 1
        self._max_speed = 0.5
        self._action_cost_coeff = 0.1
        self._hit_penalty_coeff = 1.
        self._goal_distance_cost_coeff = 0.1
        self._goal_bonus = 1
        self._goal_tolerance = 0.25

        if goals is None:
            if maze_id == 0:
                goals = np.array(((0, 3),))
            elif maze_id == 1:
                goals = np.array(((0, 1),))

        self._goals = goals

        if dynamics is None:
            dynamics = MazeDynamics

        self._dynamics = dynamics(maze_id=maze_id,
                                  max_force=self._max_force,
                                  max_speed=self._max_speed)

    def reset(self):
        self._dynamics.reset()
        return self.get_current_obs()

    @property
    def observation_space(self):
        return Box(
            low=np.inf,
            high=np.inf,
            shape=(4,)
        )

    @property
    def action_space(self):
        return Box(
            low=-self._max_force,
            high=self._max_force,
            shape=(2,)
        )

    def get_current_obs(self):
        return self._dynamics.state

    def step(self, action):
        action = action.flatten()

        # Advance.
        hit_speed = self._dynamics.step(action)

        # Check if we reached the goal.
        d = [np.linalg.norm(self._dynamics.position - g) for g in self._goals]
        dist_to_nearest_goal = min(d)
        done = dist_to_nearest_goal < self._goal_tolerance

        # Compute rewards / costs.
        if not done:
            action_norm_sq = np.sum(action**2)
            action_cost = self._action_cost_coeff * action_norm_sq
            goal_cost = self._goal_distance_cost_coeff * dist_to_nearest_goal
            hit_cost = self._hit_penalty_coeff * hit_speed
            reward = - action_cost - goal_cost - hit_cost
            # print(action_cost, goal_cost, hit_cost)
        else:
            reward = self._goal_bonus

        obs = self.get_current_obs()

        return Step(obs, reward, done)

    @staticmethod
    def plot_paths(paths, ax, style='b'):
        lines = list()
        for path in paths:
            xys = path['observations']  # H x 2

            xs = xys[:, 0]
            ys = xys[:, 1]
            lines += ax.plot(xs, ys, style)

        return lines

    def init_plot(self, ax):
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim((-5, 5))
        ax.set_ylim((-1, 2))
        ax.grid(True)
        for goal in self._goals:
            ax.plot(goal[0], goal[1], 'xk', mew=8, ms=16)

        self._dynamics.draw_env(ax)

    @overrides
    def log_diagnostics(self, paths):
        pass

    def get_param_values(self):
        return None

    def set_param_values(self, params):
        pass

    @overrides
    def log_diagnostics(self, paths):
        goals_reached = 0
        for path in paths:
            pos = path["observations"][-1][:2]  # 2
            dists_sq = np.sum((pos[None] - self._goals)**2, axis=1)  # G
            goals_reached = np.sum(dists_sq < self._goal_tolerance)

        stats = {
            "env:n_goals_reached": goals_reached
        }
        return stats

    def __getstate__(self):
       return dict(dynamics=self._dynamics.__getstate__())

    def __setstate__(self, d):
        self._dynamics.__setstate__(d['dynamics'])


class PointMassDynamics(object):
    def __init__(self, max_force, max_speed, mass=10):
        self._max_force = max_force
        self._max_speed = max_speed
        self._A = np.array([[1, 0, 1, 0],
                            [0, 1, 0, 1],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        self._B = np.array([[0, 0],
                            [0, 0],
                            [1./mass, 0],
                            [0, 1./mass]])

        self._init_state = np.zeros((4,))
        self._state = self._init_state
        self._max_speed_so_far = 0.0

    def reset(self):
        self._state = self._init_state

    def _clip_force(self, action):
        action_norm = np.linalg.norm(action)
        if action_norm > self._max_force:
            action = action / action_norm * self._max_force

        return action

    def _clip_speed(self, velocity):
        speed = np.linalg.norm(velocity)
        if speed > self._max_speed_so_far:
            self._max_speed_so_far = speed
        if speed > self._max_speed:
            velocity = velocity / speed * self._max_speed
        return velocity

    def step(self, action):
        action = self._clip_force(action)
        self._state = self._A.dot(self._state) + self._B.dot(action)

        self._state[2:] = self._clip_speed(self._state[2:])

    def draw_env(self, ax):
        pass

    @property
    def state(self):
        return self._state.copy()

    @property
    def velocity(self):
        return self._state[2:]

    @property
    def position(self):
        return self._state[:2]

    def __getstate__(self):
        d = dict(state=self._state)
        return d

    def __setstate__(self, d):
        self._state = d['state']


class MazeDynamics(PointMassDynamics):
    def __init__(self, maze_id=0, *args, **kwargs):
        super(MazeDynamics, self).__init__(*args, **kwargs)

        self._eps = 1e-3

        if maze_id == 0:
            # Horizontal walls (y-coord, x_start, x_end)
            self._h_walls = np.array([[4.0, -4, 4],
                                      [1.0, -1, 1],
                                      [-1.0, -4, 4]])
            # Vertical walls (x-coord, y_start, y_end)
            self._v_walls = np.array([[-4, -1, 4],
                                      # [-0.5, -0.5, 0.5],
                                      [4, -1, 4]])

        elif maze_id == 1:
            # Horizontal walls (y-coord, x_start, x_end)
            self._h_walls = np.array([[1.5, -4, 4],
                                      [0.5, -3, 3],
                                      [-0.5, -4, 4]])
            # Vertical walls (x-coord, y_start, y_end)
            self._v_walls = np.array([[-4, -0.5, 1.5],
                                      # [-0.5, -0.5, 0.5],
                                      [4, -0.5, 1.5]])
        else:
            raise ValueError

    def step(self, action):
        action = self._clip_force(action)
        next_state = self._A.dot(self._state) + self._B.dot(action)

        # Check if next state violates any constraints (walls).
        xp, yp = self._state[:2]
        xn, yn, dxn, dyn = next_state

        hit_speed = 0.
        for y, x_start, x_end in self._h_walls:
            if x_start < xn < x_end or x_start < xp < x_end:
                if yp > y > yn:
                    yn = y + self._eps
                    hit_speed = abs(dyn)
                    dyn = 0
                elif yp < y < yn:
                    yn = y - self._eps
                    hit_speed = abs(dyn)
                    dyn = 0

        for x, y_start, y_end in self._v_walls:
            if y_start < yn < y_end or y_start < yp < y_end:
                if xp < x < xn:
                    xn = x - self._eps
                    hit_speed = abs(dxn)
                    dxn = 0
                elif xp > x > xn:
                    xn = x + self._eps
                    hit_speed = abs(dxn)
                    dxn = 0

        self._state = np.array((xn, yn, dxn, dyn))
        self._state[2:] = self._clip_speed(self._state[2:])
        return hit_speed

    @overrides
    def draw_env(self, ax):
        for y, x_start, x_end in self._h_walls:
            ax.plot((x_start, x_end), (y, y), 'k', linewidth=2)
        for x, y_start, y_end in self._v_walls:
            ax.plot((x, x), (y_start, y_end), 'k', linewidth=2)

