""" Gaussian mixture policy. """

from contextlib import contextmanager
import numpy as np
import tensorflow as tf

from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.core.serializable import Serializable

from sac.distributions.gmm import GMM
from sac.policies.nn_policy import NNPolicy
from sac.misc import tf_utils


class GMMPolicy(NNPolicy, Serializable):
    def __init__(self, env_spec, K=2, hidden_layers=(100, 100), reg=0.001,
                 squash=True, qf=None):
        Serializable.quick_init(self, locals())

        self._hidden_layers = hidden_layers
        self._Da = env_spec.action_space.flat_dim
        self._Ds = env_spec.observation_space.flat_dim
        self._K = K
        self._is_deterministic = False
        self._qf = qf
        self._reg = reg

        self._obs_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Ds],
            name='observation',
        )

        self._dist = self.get_distribution_for(self._obs_pl)

        super(GMMPolicy, self).__init__(
            env_spec,
            self._obs_pl,
            tf.tanh(self._dist.x_t) if squash else self._dist.x_t,
            'policy'
        )

    def get_distribution_for(self, obs_t, reuse=False):
        with tf.variable_scope('policy', reuse=reuse):
            gmm = GMM(
                K=self._K,
                hidden_layers=self._hidden_layers,
                Dx=self._Da,
                cond_t_lst=[obs_t],
                reg=self._reg
            )

        return gmm

    @overrides
    def get_action(self, obs):
        if not self._is_deterministic:
            return NNPolicy.get_action(self, obs)

        # Handle the deterministic case separately.
        if self._qf is None:
            raise AttributeError

        # Get first the GMM means.
        feeds = {self._obs_pl: obs[None]}
        mus = tf.get_default_session().run(self._dist.mus_t, feeds)[0]  # K x Da

        qs = self._qf.eval(obs[None], mus)
        max_ind = np.argmax(qs)

        return mus[max_ind, :], {}  # Da

    @contextmanager
    def deterministic(self, set_deterministic=True):
        current = self._is_deterministic
        self._is_deterministic = set_deterministic
        yield
        self._is_deterministic = current

    def log_diagnostics(self, batch):
        feeds = {self._obs_pl: batch['observations']}
        sess = tf_utils.get_default_session()
        mus, log_sigs, log_ws = sess.run(
            [self._dist.mus_t, self._dist.log_sigs_t, self._dist.log_ws_t],
            feeds
        )

        logger.record_tabular('gmm-mus-mean', np.mean(mus))
        logger.record_tabular('gmm-mus-min', np.min(mus))
        logger.record_tabular('gmm-mus-max', np.max(mus))
        logger.record_tabular('gmm-mus-std', np.std(mus))
        logger.record_tabular('gmm-log-w-mean', np.mean(log_ws))
        logger.record_tabular('gmm-log-w-min', np.min(log_ws))
        logger.record_tabular('gmm-log-w-max', np.max(log_ws))
        logger.record_tabular('gmm-log-w-std', np.std(log_ws))
        logger.record_tabular('gmm-log-sigs-mean', np.mean(log_sigs))
        logger.record_tabular('gmm-log-sigs-min', np.min(log_sigs))
        logger.record_tabular('gmm-log-sigs-max', np.max(log_sigs))
        logger.record_tabular('gmm-log-sigs-std', np.std(log_sigs))
