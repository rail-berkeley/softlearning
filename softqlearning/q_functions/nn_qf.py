import tensorflow as tf
import numpy as np
from rllab.q_functions.base import QFunction
from rllab.misc.overrides import overrides


class NNQFunction(QFunction):
    def __init__(self, obs_pl, actions_pl, output, net):
        super().__init__()
        self._obs_pl = obs_pl
        self._actions_pl = actions_pl
        self._output = output
        self._net = net

    def plot_level_curves(self, ax_lst, observations, action_dims, xlim, ylim):
        """ Plots level curves of critic output.

        :param ax_lst: list of plt Axes instances.
        :param observations: list of input observations at which the Q-function
            is evaluated.
        :param action_dims: Which two action dimensions are varied (other
            dimensions are set to zero).
        :param xlim: Minimum and maximum values along the first action dimension
            (x_min, x_max).
        :param ylim: Minimum and maximum values along the second action
            dimensions (y_min, y_max).
        :return:
        """
        """

        :param ax_lst: List of plt Axes instances.
        :param obs_lst: List of input observations at which the critic is
            evaluated.
        :return:
        """
        assert len(action_dims) == 2

        Da = self._actions_pl.get_shape()[-1].value
        xx = np.arange(xlim[0], xlim[1], 0.05)
        yy = np.arange(ylim[0], ylim[1], 0.05)
        X, Y = np.meshgrid(xx, yy)

        actions = np.zeros((X.size, Da))
        actions[:, action_dims[0]] = X.ravel()
        actions[:, action_dims[1]] = Y.ravel()

        feed = {self._obs_pl: observations,
                self._actions_pl: actions}

        Q = tf.get_default_session().run(self._output, feed)
        Q = Q.reshape((-1,) + X.shape)

        for ax, qs in zip(ax_lst, Q):
            cs = ax.contour(X, Y, qs, 20)
            ax.clabel(cs, inline=1, fontsize=10, fmt='%.2f')
            ax.grid(True)

    @overrides
    def get_params_internal(self, **tags):
        return self._net.get_params_internal(**tags)
