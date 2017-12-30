import time
from unittest import mock

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from sac.policies import RealNVPPolicy


def generate_grid_data(x_min=-1, x_max=1, y_min=-1, y_max=1, nx=5, ny=5, density=200):
    xx = np.linspace(x_min, x_max, nx)
    yy = np.linspace(y_min, y_max, ny)
    xs, ys = [], []
    for x in xx:
        xs += (np.ones(density) * x).tolist()
        ys += np.linspace(min(yy), max(yy), density).tolist()
    for y in yy:
        ys += (np.ones(density) * y).tolist()
        xs += np.linspace(min(xx), max(xx), density).tolist()
    return np.array([xs, ys]).swapaxes(0, 1)


def _log_gaussian_2d(mean, variance, inputs):
    weighted_sq_diff = (inputs - mean[None])**2 / variance[None]
    weighted_sq_sum = tf.reduce_sum(weighted_sq_diff, axis=1,
                                    keep_dims=True)
    log_Z = - tf.reduce_sum(0.5 * tf.log(variance)) - tf.log(2 * np.pi)
    log_gauss = log_Z - weighted_sq_sum / 2

    return log_gauss  # N x 1


def _log_target(weights, means, variances, conditions, inputs):
    # inputs: N x Dx
    weights, means, variances = [
        v.astype(np.float32) for v in [ weights, means, variances ]
    ]
    target_log_components = [
        _log_gaussian_2d(mean, variance, inputs) + tf.log(weight)
        for weight, mean, variance in zip(weights, means, variances)
    ]

    target_log_components = tf.concat(target_log_components, axis=1)
    target_log_components -= (1 - conditions) * 99

    return tf.reduce_logsumexp(target_log_components, axis=1)  # N


class RealNVP2dRlExample(object):
    def __init__(self,
                 plt_subplots,
                 weights, means, variances,
                 policy_config=None,
                 seed=None,
                 batch_size=64,
                 num_epochs=1000,
                 num_steps_per_epoch=100):
        self.weights, self.means, self.variances = weights, means, variances
        self.fig, self.ax = plt_subplots
        self.cs = None

        if seed is not None:
            print('Seed: ' + str(seed))
            tf.set_random_seed(seed)
            np.random.seed(seed)


        self.num_epochs = num_epochs
        self.num_steps_per_epoch = num_steps_per_epoch
        self._batch_size = batch_size


        D_in = 2
        env_spec = mock.Mock()
        env_spec.action_space.flat_dim = D_in
        env_spec.observation_space.flat_dim = D_in

        def create_target_wrapper(weights, means, variances):
            def wraps(observations, actions):
                return _log_target(
                    weights, means, variances, observations, actions)
            return wraps

        self.policy = RealNVPPolicy(
            env_spec=env_spec,
            config=policy_config,
            qf=create_target_wrapper(weights, means, variances))

        self.policy_config = policy_config

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def run(self):
        print("epoch | loss")
        with self.session.as_default():
            print("true value: ", self._compute_true_value())
            for epoch in range(1, self.num_epochs + 1):
                for i in range(1, self.num_steps_per_epoch + 1):
                    observations = np.random.randint(
                        0, 2, size=(self._batch_size, 2))
                    _, loss = self.session.run(
                        (self.policy.train_op, self.policy.loss),
                        feed_dict={
                            self.policy._observations_ph: observations
                        }
                    )

                fixed_observation = np.random.randint(0, 2, size=(1, 2))
                self.update_figure_texts(fixed_observation)
                self.redraw_samples(fixed_observation)
                self.redraw_contours(fixed_observation)
                self.fig.canvas.draw()
                print("{epoch:05d} | {loss:.5f}".format(
                    epoch=epoch, loss=loss))

    def update_figure_texts(self, fixed_observation):
        fixed_observation_text = "fixed_observation: " + str(fixed_observation)
        if not getattr(self, "fixed_observation_text", None):
            self.fixed_observation_text = self.ax.text(
                -2, 2, fixed_observation_text, fontsize=15)
        else:
            self.fixed_observation_text.set_text(fixed_observation_text)

    def redraw_samples(self, fixed_observation):
        observations = np.tile(fixed_observation, (self._batch_size, 1))
        sampled_x, sampled_y = self.session.run(
            (self.policy.x, self.policy._action),
            feed_dict={self.policy._observations_ph: observations}
        )

        x_grid = generate_grid_data(-2.0, 2.0, -2.0, 2.0, 20, 20)
        observations = np.tile(fixed_observation, (x_grid.shape[0], 1))
        y_grid = self.session.run(
            self.policy._action, feed_dict={
                self.policy.x: x_grid,
                self.policy._observations_ph: observations
            }
        )

        if not getattr(self, "samples_lines", None):
            self.samples_lines = self.ax.plot(
                sampled_y[:, 0], sampled_y[:, 1], 'bx')[0]
            self.samples_x_lines = self.ax.plot(
                sampled_x[:, 0], sampled_x[:, 1], 'rx')[0]
            self.y_grid_lines = self.ax.plot(
                y_grid[:, 0], y_grid[:, 1], 'k.', markersize=0.25)[0]
        else:
            self.samples_lines.set_data(sampled_y[:, 0], sampled_y[:, 1])
            self.samples_x_lines.set_data(sampled_x[:, 0], sampled_x[:, 1])
            self.y_grid_lines.set_data(y_grid[:, 0], y_grid[:, 1])

    def redraw_contours(self, fixed_observation):
        MIN, MAX = -2.1, 2.1
        xs = np.linspace(MIN, MAX, 100)
        ys = np.linspace(MIN, MAX, 100)
        mesh_x, mesh_y = np.meshgrid(xs, ys)

        y = np.stack((mesh_x.ravel(), mesh_y.ravel()),
                     axis=1).astype(np.float32)

        if self.policy_config["squash"]:
            y = np.arctanh(y)

        observations = np.tile(fixed_observation, (y.shape[0], 1))
        p_y, Q = self.session.run(
            (self.policy.pi, tf.exp(self.policy.Q)), feed_dict={
                self.policy.y: y,
                self.policy._observations_ph: observations
            }
        )
        p_y = p_y.reshape(mesh_x.shape)
        Q = Q.reshape(mesh_x.shape)

        levels = 20
        cmap = plt.cm.viridis

        if self.cs is None:
            self.ax.plot(self.means[:, 0],
                         self.means[:, 1], 'kx', markersize=20)
            self.Q_contour = self.ax.contour(mesh_x, mesh_y, Q, 30,
                                             cmap=cmap, linewidths=0.5)
            self.cs = self.ax.contour(mesh_x, mesh_y, p_y, levels, cmap=cmap)
        else:
            for tp in self.cs.collections:
                tp.remove()
            for tp in self.Q_contour.collections:
                tp.remove()
            self.Q_contour = self.ax.contour(
                mesh_x, mesh_y, Q, 30, cmap=cmap, linewidths=0.5)
            self.cs = self.ax.contour(mesh_x, mesh_y, p_y, levels, cmap=cmap)

    def _compute_true_value(self):
        MIN = -0.999
        MAX = 0.999

        xs = np.linspace(MIN, MAX, 1000)
        ys = np.linspace(MIN, MAX, 1000)
        X, Y = np.meshgrid(xs, ys)

        x = np.stack((X.ravel(), Y.ravel()), axis=1)
        values = self.session.run(
            tf.exp(self.policy.Q),
            feed_dict={
                self.policy._action: x,
                self.policy._observations_ph: np.ones((x.shape[0], 2))
            }
        )

        values = values.reshape(X.shape)

        da = (xs[1] - xs[0])**2
        V = np.log(np.sum(values * da))
        return V


if __name__ == '__main__':
    example = RealNVP2dRlExample()
    example.run()
