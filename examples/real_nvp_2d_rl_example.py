import time
from unittest import mock

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from sac.policies.real_nvp import RealNVPPolicy

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

def _log_target(weights, means, variances, inputs):
  # inputs: N x Dx
  weights, means, variances = [v.astype(np.float32) for v in [weights, means, variances]]
  target_log_components = [
    _log_gaussian_2d(mean, variance, inputs) + tf.log(weight)
    for weight, mean, variance in zip(weights, means, variances)
  ]

  target_log_components = tf.concat(target_log_components, axis=1)

  return tf.reduce_logsumexp(target_log_components, axis=1)  # N

class RealNVP2dRlExample(object):
  def __init__(self,
               plt_subplots,
               weights, means, variances,
               policy_config=None,
               seed=None):
    self.weights, self.means, self.variances = weights, means, variances
    self.fig, self.ax = plt_subplots
    self.cs = None

    if seed is not None:
      print('Seed: ' + str(seed))
      tf.set_random_seed(seed)
      np.random.seed(seed)

    self._batch_size = 64

    def create_target_wrapper(weights, means, variances):
        def wraps(inputs):
            return _log_target(weights, means, variances, inputs)
        return wraps

    env_spec = mock.Mock()
    env_spec.action_space.flat_dim = 2
    env_spec.observation_space.flat_dim = 2

    self.policy = RealNVPPolicy(
        env_spec=env_spec,
        config=policy_config,
        qf=create_target_wrapper(weights, means, variances))

    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())

  def run(self):
    NUM_EPOCHS, NUM_STEPS = 100, 100

    print("epoch | loss")
    with self.session.as_default():
      print("true value: ", self._compute_true_value())
      for epoch in range(1, NUM_EPOCHS+1):
        for i in range(1, NUM_STEPS+1):
          _, loss = self.session.run(
            (self.policy.train_op, self.policy.loss),
            feed_dict={ self.policy.batch_size: self._batch_size, }
          )

        self.redraw_samples()
        self.redraw_contours()
        print("{epoch:05d} | {loss:.5f}".format(
          epoch=epoch, loss=loss))

  def redraw_samples(self):
      sampled_x, sampled_y  = self.session.run(
          (self.policy.x, self.policy.y),
          feed_dict={ self.policy.batch_size: self._batch_size }
      )
      x_grid = generate_grid_data(-2.0, 2.0, -2.0, 2.0, 20, 20)
      y_grid = self.session.run(
          self.policy.y, feed_dict={ self.policy.x: x_grid }
      )

      if not getattr(self, "samples_lines", None):
          self.samples_lines = self.ax.plot(sampled_y[:, 0], sampled_y[:, 1], 'bx')[0]
          self.samples_x_lines = self.ax.plot(sampled_x[:, 0], sampled_x[:, 1], 'rx')[0]
          self.y_grid_lines = self.ax.plot(y_grid[:, 0], y_grid[:, 1], 'k.', markersize=0.25)[0]
      else:
          self.samples_lines.set_data(sampled_y[:, 0], sampled_y[:, 1])
          self.samples_x_lines.set_data(sampled_x[:, 0], sampled_x[:, 1])
          self.y_grid_lines.set_data(y_grid[:, 0], y_grid[:, 1])

  def redraw_contours(self):
    MIN, MAX = -2.1, 2.1
    xs = np.linspace(MIN, MAX, 100)
    ys = np.linspace(MIN, MAX, 100)
    mesh_x, mesh_y = np.meshgrid(xs, ys)

    y = np.stack((mesh_x.ravel(), mesh_y.ravel()), axis=1).astype(np.float32)
    p_y = self.session.run(
        self.policy.pi, feed_dict={ self.policy.y: y }
    ).reshape(mesh_x.shape)

    levels = 20
    cmap = plt.cm.viridis

    if self.cs is None:
      self.ax.plot(self.means[:, 0], self.means[:, 1], 'kx', markersize=20)
      Q = self.session.run(
          tf.exp(self.policy.Q), feed_dict={ self.policy.y: y }
      ).reshape(mesh_x.shape)
      self.ax.contour(mesh_x, mesh_y, Q, 30, colors="silver", linewidths=0.5)
      self.cs = self.ax.contour(mesh_x, mesh_y, p_y, levels, cmap=cmap)
    else:
      for tp in self.cs.collections:
        tp.remove()
      self.cs = self.ax.contour(mesh_x, mesh_y, p_y, levels, cmap=cmap)

    self.fig.canvas.draw()

  def _compute_true_value(self):
    MIN = -0.999
    MAX = 0.999

    xs = np.linspace(MIN, MAX, 1000)
    ys = np.linspace(MIN, MAX, 1000)
    X, Y = np.meshgrid(xs, ys)

    x = np.stack((X.ravel(), Y.ravel()), axis=1)
    values = self.session.run(
        tf.exp(self.policy.Q),
        feed_dict={ self.policy.y: x }
    )

    values = values.reshape(X.shape)

    da = (xs[1] - xs[0])**2
    V = np.log(np.sum(values[1:, 1:]*da))
    return V

if __name__ == '__main__':
  example = RealNVP2dRlExample()
  example.run()
