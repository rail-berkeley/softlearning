import time

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from sac.distributions.real_nvp import RealNVP

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
               seed=None):
    self.weights, self.means, self.variances = weights, means, variances
    self.fig, self.ax = plt_subplots
    self.cs = None

    if seed is not None:
      print('Seed: ' + str(seed))
      tf.set_random_seed(seed)
      np.random.seed(seed)

    self._batch_size = 64

    nvp_config = {
      "mode": "train",
      "D_in": 2,
      "learning_rate": 5e-4,
      "scale_regularization": 0.0, # 5e-2,
      "num_coupling_layers": 8,
      "translation_hidden_sizes": (25,),
      "scale_hidden_sizes": (25,),
      "squash": False
    }

    tf.reset_default_graph()

    def create_target_wrapper(weights, means, variances):
        def wraps(inputs):
            return _log_target(weights, means, variances, inputs)
        return wraps

    self.policy_distribution = RealNVP(
        config=nvp_config,
        create_target_fn=create_target_wrapper(weights, means, variances))

    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())

  def run(self):
    NUM_EPOCHS, NUM_STEPS = 100, 100

    print("epoch | forward_loss")
    with self.session.as_default():
      for epoch in range(1, NUM_EPOCHS+1):
        for i in range(1, NUM_STEPS+1):
          _, sampled_z, forward_loss = self.session.run(
            (
              self.policy_distribution.train_op,
              self.policy_distribution.z,
              self.policy_distribution.forward_loss,
            ),
            feed_dict={
              self.policy_distribution.batch_size: self._batch_size,
            }
          )

        if i % 10 == 0:
            self.redraw_samples(sampled_z)

        print("{epoch:05d} | {forward_loss:.5f}".format(
          epoch=epoch, forward_loss=forward_loss))

        self.redraw_contours()

  def redraw_samples(self, sampled_z):
      if not getattr(self, "samples_lines", None):
          self.samples_lines = self.ax.plot(sampled_z[:, 0], sampled_z[:, 1], 'bx')[0]
      else:
          self.samples_lines.set_data(sampled_z[:, 0], sampled_z[:, 1])

  def redraw_contours(self):
    MIN, MAX = -2.1, 2.1
    xs = np.linspace(MIN, MAX, 100)
    ys = np.linspace(MIN, MAX, 100)
    mesh_x, mesh_y = np.meshgrid(xs, ys)

    x = np.stack((mesh_x.ravel(), mesh_y.ravel()), axis=1)
    log_p_z = self.session.run(
      self.policy_distribution.log_p_z,
      feed_dict={self.policy_distribution.x_placeholder: x}
    ).reshape(mesh_x.shape)
    p_z = np.exp(log_p_z)

    levels = 20
    cmap=plt.cm.viridis

    if self.cs is None:
      self.ax.plot(self.means[:, 0], self.means[:, 1], 'kx', markersize=20)
      self.cs = self.ax.contour(mesh_x, mesh_y, p_z, levels, cmap=cmap)
    else:
      for tp in self.cs.collections:
        tp.remove()
      self.cs = self.ax.contour(mesh_x, mesh_y, p_z, levels, cmap=cmap)

    self.fig.canvas.draw()

if __name__ == '__main__':
  example = RealNVP2dRlExample()
  example.run()
