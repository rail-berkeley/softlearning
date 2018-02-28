""" Gaussian mixture policy. """

from contextlib import contextmanager
import numpy as np
import tensorflow as tf

from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.core.serializable import Serializable

from sac.distributions import GMM
from sac.policies import NNPolicy
from sac.misc import tf_utils


class GMMPolicy(NNPolicy, Serializable):
    """Gaussian Mixture Model policy"""
    def __init__(self, env_spec, K=2, hidden_layer_sizes=(100, 100), reg=0.001,
                 squash=True, reparameterize=True, qf=None):
        """
        Args:
            env_spec (`rllab.EnvSpec`): Specification of the environment
                to create the policy for.
            K (`int`): Number of mixture components.
            hidden_layer_sizes (`list` of `int`): Sizes for the Multilayer
                perceptron hidden layers.
            reg (`float`): Regularization coeffiecient for the GMM parameters.
            squash (`bool`): If True, squash the GMM the gmm action samples
               between -1 and 1 with tanh.
            qf (`ValueFunction`): Q-function approximator.
        """
        Serializable.quick_init(self, locals())

        self._hidden_layers = hidden_layer_sizes
        self._Da = env_spec.action_space.flat_dim
        self._Ds = env_spec.observation_space.flat_dim
        self._K = K
        self._is_deterministic = False
        self._fixed_h = None
        self._qf = qf
        self._reg = reg
        self._reparameterize = reparameterize

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
        """Create the actual GMM distribution instance."""

        with tf.variable_scope('policy', reuse=reuse):
            gmm = GMM(
                K=self._K,
                hidden_layers_sizes=self._hidden_layers,
                Dx=self._Da,
                cond_t_lst=[obs_t],
                reg=self._reg,
                reparameterize=self._reparameterize
            )

        return gmm

    @overrides
    def get_action(self, obs):
        """Sample action based on the observations.

        If `self._is_deterministic` is True, returns a greedily sampled action
        for the observations. If False, return stochastically sampled action.
        """

        if not self._is_deterministic:
            return NNPolicy.get_action(self, obs)

        # Handle the deterministic case separately.
        if self._qf is None:
            raise AttributeError

        # Get first the GMM means.
        feeds = {self._obs_pl: obs[None]}
        mus = tf.get_default_session().run(self._dist.mus_t, feeds)[0]  # K x Da

        qs = self._qf.eval(obs[None], mus)

        if self._fixed_h is not None:
            h = self._fixed_h
        else:
            h = np.argmax(qs)

        return mus[h, :], {}  # Da

    @contextmanager
    def fix_h(self, h):
        was_deterministic = self._is_deterministic
        self._is_deterministic = True
        self._fixed_h = h
        yield
        self._fixed_h = None
        self._is_deterministic = was_deterministic

    @contextmanager
    def deterministic(self, set_deterministic=True):
        """Context manager for changing the determinism of the policy.

        See `self.get_action` for further information about the effect of
        self._is_deterministic.

        Args:
            set_deterministic (`bool`): Value to set the self._is_deterministic
            to during the context. The value will be reset back to the previous
            value when the context exits.
        """
        current = self._is_deterministic
        self._is_deterministic = set_deterministic
        yield
        self._is_deterministic = was_deterministic

    def log_diagnostics(self, iteration, batch):
        """Record diagnostic information to the logger.

        Records the mean, min, max, and standard deviation of the GMM
        means, component weights, and covariances.
        """

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
