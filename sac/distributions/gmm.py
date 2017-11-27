""" Gaussian mixture model. """

import tensorflow as tf
import numpy as np

from sac.misc.mlp import mlp

LOG_SIG_CAP_MAX = 2
LOG_SIG_CAP_MIN = -20


class GMM(object):
    def __init__(
            self,
            K,
            Dx,
            hidden_layers=(100, 100),
            reg=0.001,
            cond_t_lst=(),
    ):
        self._cond_t_lst = cond_t_lst
        self._reg = reg
        self._layer_sizes = list(hidden_layers) + [K*(2*Dx + 1)]

        self._Dx = Dx
        self._K = K

        self._create_placeholders()
        self._create_graph()

    def _create_placeholders(self):
        self._N_pl = tf.placeholder(
            tf.int32,
            shape=(),
            name='N',
        )

    @staticmethod
    def _create_log_gaussian(mu_t, log_sig_t, t):
        normalized_dist_t = (t - mu_t) * tf.exp(-log_sig_t)  # ... x D
        quadratic = - 0.5 * tf.reduce_sum(normalized_dist_t ** 2, axis=-1)
        # ... x (None)

        log_z = tf.reduce_sum(log_sig_t, axis=-1)  # ... x (None)
        D_t = tf.cast(tf.shape(mu_t)[-1], tf.float32)
        log_z += 0.5 * D_t * np.log(2 * np.pi)

        log_p = quadratic - log_z

        return log_p  # ... x (None)

    def _create_p_xz_params(self):
        K = self._K
        Dx = self._Dx

        if len(self._cond_t_lst) == 0:
            w_and_mu_and_logsig_t = tf.get_variable(
                'params', self._layer_sizes[-1],
                initializer=tf.random_normal_initializer(0, 0.1)
            )

        else:
            w_and_mu_and_logsig_t = mlp(
                inputs=self._cond_t_lst,
                layer_sizes=self._layer_sizes,
                output_nonlinearity=None,
            )  # ... x K*Dx*2+K

        w_and_mu_and_logsig_t = tf.reshape(
            w_and_mu_and_logsig_t, shape=(-1, K, 2*Dx+1))

        log_w_t = w_and_mu_and_logsig_t[..., 0]
        mu_t = w_and_mu_and_logsig_t[..., 1:1+Dx]
        log_sig_t = w_and_mu_and_logsig_t[..., 1+Dx:]

        log_sig_t = tf.clip_by_value(
            log_sig_t, LOG_SIG_CAP_MIN, LOG_SIG_CAP_MAX)

        return log_w_t, mu_t, log_sig_t

    def _create_graph(self):
        Dx = self._Dx

        if len(self._cond_t_lst) > 0:
            N_t = tf.shape(self._cond_t_lst[0])[0]
        else:
            N_t = self._N_pl

        K = self._K

        # Create p(x|z).
        with tf.variable_scope('p'):
            log_ws_t, xz_mus_t, xz_log_sigs_t = self._create_p_xz_params()
            # (N x K), (N x K x Dx), (N x K x Dx)
            xz_sigs_t = tf.exp(xz_log_sigs_t)

            # Sample the latent code.
            z_t = tf.multinomial(logits=log_ws_t, num_samples=1)  # N x 1

            # Choose mixture component corresponding to the latent.
            mask_t = tf.one_hot(
                z_t[:, 0], depth=K, dtype=tf.bool,
                on_value=True, off_value=False
            )
            xz_mu_t = tf.boolean_mask(xz_mus_t, mask_t)  # N x Dx
            xz_sig_t = tf.boolean_mask(xz_sigs_t, mask_t)  # N x Dx

            # Sample x.
            x_t = tf.stop_gradient(
                xz_mu_t + xz_sig_t * tf.random_normal((N_t, Dx))
            )  # N x Dx

            # log p(x|z)
            log_p_xz_t = self._create_log_gaussian(
                xz_mus_t, xz_log_sigs_t, x_t[:, None, :]
            )  # N x K

            # log p(x)
            log_p_x_t = tf.reduce_logsumexp(log_p_xz_t + log_ws_t, axis=1)
            log_p_x_t -= tf.reduce_logsumexp(log_ws_t, axis=1)  # N

        reg_loss_t = 0
        reg_loss_t += self._reg * 0.5 * tf.reduce_mean(xz_log_sigs_t ** 2)
        reg_loss_t += self._reg * 0.5 * tf.reduce_mean(xz_mus_t ** 2)

        self._log_p_x_t = log_p_x_t
        self._reg_loss_t = reg_loss_t
        self._x_t = x_t

        self._log_ws_t = log_ws_t
        self._mus_t = xz_mus_t
        self._log_sigs_t = xz_log_sigs_t

    @property
    def log_p_t(self):
        return self._log_p_x_t

    @property
    def reg_loss_t(self):
        return self._reg_loss_t

    @property
    def x_t(self):
        return self._x_t

    @property
    def mus_t(self):
        return self._mus_t

    @property
    def log_sigs_t(self):
        return self._log_sigs_t

    @property
    def log_ws_t(self):
        return self._log_ws_t

    @property
    def N_t(self):
        return self._N_pl
