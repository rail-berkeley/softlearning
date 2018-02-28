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
        self._squash = squash
        self._qf = qf
        self._reg = reg
        self._reparameterize = reparameterize

        self.name = name
        self.build()

        self._scope_name = (
            tf.get_variable_scope().name + "/" + name
        ).lstrip("/")
        super(NNPolicy, self).__init__(env_spec)

    def actions_for(self, observations, latents=None,
                    name=None, reuse=tf.AUTO_REUSE,
                    with_log_pis=False, regularize=False):
        name = name or self.name

        # with tf.variable_scope('policy_distribution', reuse=reuse):
        with tf.variable_scope(name, reuse=reuse):
            distribution = GMM(
                K=self._K,
                hidden_layers_sizes=self._hidden_layers,
                Dx=self._Da,
                cond_t_lst=(observations,),
                reg=self._reg
            )

        raw_actions = tf.stop_gradient(distribution.x_t)
        actions = tf.tanh(raw_actions) if self._squash else raw_actions

        # TODO: should always return same shape out
        # Figure out how to make the interface for `log_pis` cleaner
        if with_log_pis:
            # TODO.code_consolidation: should come from log_pis_for
            log_pis = distribution.log_p_t
            return actions, log_pis

        return actions

    def build(self):
        self._observations_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self._Ds),
            name='observations',
        )

        self._latents_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self._Da),
            name='latents',
        )

        # TODO.code_consolidation:
        # self.distribution is used very differently compared to the
        # `RealNVPPolicy`s distribution.
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.distribution = GMM(
                K=self._K,
                hidden_layers_sizes=self._hidden_layers,
                Dx=self._Da,
                cond_t_lst=[obs_t],
                reg=self._reg,
                reparameterize=self._reparameterize
            )

        raw_actions = tf.stop_gradient(self.distribution.x_t)
        self._actions = tf.tanh(raw_actions) if self._squash else raw_actions
        # TODO.code_consolidation:
        # This should be standardized with RealNVPPolicy/NNPolicy
        # self._determistic_actions = self.actions_for(self._observations_ph,
        #                                              self._latents_ph)

    @overrides
    def get_action(self, observation):
        """Sample single action based on the observations.

        TODO: Modify `NNPolicy.get_action` and remove this
        """
        return self.get_actions(observation[None])[0], {}

    @overrides
    def get_actions(self, obs):
        """Sample actions based on the observations.

        If `self._is_deterministic` is True, returns a greedily sampled action
        for the observations. If False, return stochastically sampled action.
        """
        feed_dict = {self._observations_ph: obs}

        if not self._is_deterministic:
            actions = tf.get_default_session().run(self._actions, feed_dict)
            return actions

        # Handle the deterministic case separately.
        if self._qf is None:
            raise AttributeError

        # TODO.code_consolidation: these shapes should be fixed
        mus = tf.get_default_session().run(
            self.distribution.mus_t, feed_dict)[0]  # K x Da

        qs = self._qf.eval(obs, mus)[:, None]

        if self._fixed_h is not None:
            h = self._fixed_h # TODO.code_consolidation: this needs to be tiled
        else:
            h = np.argmax(qs, axis=1) # TODO.code_consolidation: check the axis

        return mus[h, :]  # Da

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

        feeds = {self._observations_ph: batch['observations']}
        sess = tf_utils.get_default_session()
        mus, log_sigs, log_ws = sess.run(
            (
                self.distribution.mus_t,
                self.distribution.log_sigs_t,
                self.distribution.log_ws_t,
            ),
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
