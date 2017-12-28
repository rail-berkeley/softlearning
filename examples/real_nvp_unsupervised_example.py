import time

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from sac.distributions.real_nvp_v2 import RealNVP

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

class RealNVP2UnsupervisedlExample(object):
  def __init__(self,
               x_train,
               subplots,
               real_nvp_config=None,
               seed=None,
               batch_size=64,
               plot_every=10):
    if seed is not None:
      print('Seed: ' + str(seed))
      tf.set_random_seed(seed)
      np.random.seed(seed)

    self.x_train = x_train
    self.subplots = subplots
    self.ax_markers = {
        "X_grid": None,
        "X_samples": None,
        "Z_grid": None,
        "Z_samples": None
    }

    self.plot_every = plot_every
    self.x_grid = generate_grid_data(-1.5, 2.5, -1.0, 1.5, 20, 20)

    self.batch_size = batch_size

    self.real_nvp_config = real_nvp_config
    self.real_nvp = RealNVP(config=real_nvp_config)

    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())

  def run(self):
    NUM_EPOCHS = 100

    N_train = self.x_train.shape[0]
    print("epoch | loss")
    with self.session.as_default():
      for epoch in range(1, NUM_EPOCHS+1):
        num_steps = (N_train // self.batch_size) + 1
        for i in range(1, num_steps+1):
          batch_idx = np.random.choice(N_train,
                                       self.batch_size,
                                       replace=False)
          x_batch = self.x_train[batch_idx, :]

          _, loss = self.session.run(
            (self.real_nvp.train_op, self.real_nvp.loss),
            feed_dict={self.real_nvp.x_placeholder: x_batch}
          )

          if i % self.plot_every == 0:
              self.redraw_plots()

        print("{epoch:05d} | {loss:.5f}".format(
          epoch=epoch, loss=loss))

  def redraw_plots(self):
      figs, axs = self.subplots
      grid_markersize = 0.5
      sample_markersize = None

      N_train = self.x_train.shape[0]
      samples_idx = np.random.choice(N_train,
                                     1000,
                                     replace=False)
      x_samples = self.x_train[samples_idx, :]
      x_grid = self.x_grid

      z_samples = self.session.run(
          self.real_nvp.z,
          feed_dict={self.real_nvp.x_placeholder: x_samples}
      )
      z_grid = self.session.run(
          self.real_nvp.z,
          feed_dict={self.real_nvp.x_placeholder: x_grid}
      )

      if self.ax_markers.get("X_grid") is None:
          self.ax_markers["X_grid"] = axs[0].plot(
              x_grid[:, 0], x_grid[:, 1],
              'k.',
              markersize=grid_markersize)[0]
      else:
          self.ax_markers["X_grid"].set_data(
              x_grid[:, 0], x_grid[:, 1])

      if self.ax_markers.get("X_samples") is None:
          self.ax_markers["X_samples"] = axs[0].plot(
              x_samples[:, 0], x_samples[:, 1],
              'b.',
              markersize=sample_markersize)[0]
      else:
          self.ax_markers["X_samples"].set_data(
              x_samples[:, 0], x_samples[:, 1])

      if self.ax_markers.get("Z_grid") is None:
          self.ax_markers["Z_grid"] = axs[1].plot(
              z_grid[:, 0], z_grid[:, 1],
              'k.',
              markersize=grid_markersize)[0]
      else:
          self.ax_markers["Z_grid"].set_data(
              z_grid[:, 0], z_grid[:, 1])

      if self.ax_markers.get("Z_samples") is None:
          self.ax_markers["Z_samples"] = axs[1].plot(
              z_samples[:, 0], z_samples[:, 1],
              'b.',
              markersize=sample_markersize)[0]
      else:
          self.ax_markers["Z_samples"].set_data(
              z_samples[:, 0], z_samples[:, 1])

      figs.canvas.draw()
